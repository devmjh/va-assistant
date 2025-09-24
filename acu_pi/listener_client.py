# filename: acu_pi/listener_client.py
import grpc
import sounddevice as sd
from pocketsphinx import Config, Decoder
import webrtcvad
import numpy as np
import collections
import sys
import os

# --- Add protos directory to path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'protos'))

import audiostream_pb2
import audiostream_pb2_grpc

# --- Configuration ---
BRAIN_ADDRESS = 'rdunano2:50051'
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 30
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
VAD_AGGRESSIVENESS = 1
SILENCE_CHUNKS_TRIGGER = 50
PRE_SPEECH_BUFFER_CHUNKS = 10
WAKE_WORD = "bridge to engineering"

def main():
    """Main loop: opens one audio stream and uses a state machine
       to switch between wake-word detection and command streaming."""
    
    # --- Initialize PocketSphinx Decoder ---
    # This configuration points to the stable model path installed by apt.
    config = Config(
        hmm='/usr/share/pocketsphinx/model/en-us/en-us',
        dict='/usr/share/pocketsphinx/model/en-us/cmudict-en-us.dict',
        keyphrase=WAKE_WORD,
        kws_threshold=1e-20,
        logfn='/dev/null'
    )
    decoder = Decoder(config)

    # --- Initialize VAD ---
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    
    # --- State Machine Variables ---
    listening_for_command = False
    
    # --- Open ONE persistent audio stream ---
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', blocksize=CHUNK_SIZE) as stream:
            print(f"✅ ACU is running. Waiting for '{WAKE_WORD}'...")
            
            while True: # Main loop processing audio chunks
                chunk, overflowed = stream.read(CHUNK_SIZE)
                chunk_bytes = chunk.tobytes()

                if not listening_for_command:
                    # --- WAKE WORD DETECTION MODE ---
                    decoder.start_utt()
                    decoder.process_raw(chunk_bytes, False, False)
                    if decoder.hyp() is not None:
                        print(f"✅ Wake word detected!")
                        listening_for_command = True
                        
                        # Prepare for command streaming
                        pre_speech_buffer = collections.deque(maxlen=PRE_SPEECH_BUFFER_CHUNKS)
                        triggered = False
                        silence_chunks = 0
                        
                        # Start a new gRPC stream in a structured way
                        try:
                            with grpc.insecure_channel(BRAIN_ADDRESS) as channel:
                                stub = audiostream_pb2_grpc.AudioStreamerStub(channel)
                                print(f"Connecting to server at {BRAIN_ADDRESS}...")
                                
                                # This inner generator will handle the VAD logic
                                def command_audio_iterator():
                                    nonlocal listening_for_command, triggered, silence_chunks
                                    print("✅ Listening for command...")
                                    
                                    # First, yield the current chunk that triggered the wake word
                                    yield audiostream_pb2.AudioChunk(audio_chunk=chunk_bytes)
                                    
                                    # Now, read new chunks for the command
                                    while listening_for_command:
                                        cmd_chunk, _ = stream.read(CHUNK_SIZE)
                                        cmd_chunk_bytes = cmd_chunk.tobytes()
                                        
                                        if not triggered:
                                            pre_speech_buffer.append(cmd_chunk_bytes)
                                        
                                        is_speech = vad.is_speech(cmd_chunk_bytes, SAMPLE_RATE)

                                        if is_speech:
                                            if not triggered:
                                                print("Speech detected, streaming...")
                                                triggered = True
                                                for buffered_chunk in pre_speech_buffer:
                                                    yield audiostream_pb2.AudioChunk(audio_chunk=buffered_chunk)
                                                pre_speech_buffer.clear()
                                            
                                            yield audiostream_pb2.AudioChunk(audio_chunk=cmd_chunk_bytes)
                                            silence_chunks = 0
                                        elif triggered:
                                            yield audiostream_pb2.AudioChunk(audio_chunk=cmd_chunk_bytes)
                                            silence_chunks += 1
                                            if silence_chunks > SILENCE_CHUNKS_TRIGGER:
                                                print("End of command detected.")
                                                listening_for_command = False # End this command session
                                        elif not triggered and len(pre_speech_buffer) == PRE_SPEECH_BUFFER_CHUNKS:
                                            # If buffer is full and no speech, timeout
                                            print("No command heard, timing out.")
                                            listening_for_command = False
                                
                                # Make the RPC call with the generator
                                response = stub.StreamAudio(command_audio_iterator())
                                print(f"Server response: '{response.status_message}'")
                        
                        except grpc.RpcError as e:
                            print(f"gRPC stream failed: {e}")
                        finally:
                            # Reset state for the next wake word
                            listening_for_command = False
                            print(f"\n✅ ACU is running. Waiting for '{WAKE_WORD}'...")

                    decoder.end_utt()

    except Exception as e:
        print(f"An error occurred with the audio stream: {e}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.")