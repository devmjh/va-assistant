# filename: acu_pi/listener_client.py
import grpc
import sounddevice as sd
from pocketsphinx import LiveSpeech
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
BRAIN_ADDRESS = 'rdunano2:50051' # Using hostname now
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 30  # VAD supports 10, 20, or 30 ms chunks
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
VAD_AGGRESSIVENESS = 1 # 0 is least aggressive, 3 is most aggressive
SILENCE_CHUNKS_TRIGGER = 50 # 50 chunks * 30ms = 1.5 seconds of silence
PRE_SPEECH_BUFFER_CHUNKS = 10 # Buffer 10 * 30ms = 0.3 seconds of audio before speech
WAKE_WORD = "bridge to engineering"

def stream_audio_to_brain(audio_iterator):
    """Opens a gRPC channel and streams audio chunks from an iterator."""
    try:
        with grpc.insecure_channel(BRAIN_ADDRESS) as channel:
            stub = audiostream_pb2_grpc.AudioStreamerStub(channel)
            print(f"Connecting to server at {BRAIN_ADDRESS}...")
            
            # The generator function is the audio_iterator itself
            response = stub.StreamAudio(audio_iterator)
            print(f"Server response: '{response.status_message}'")
    except grpc.RpcError as e:
        print(f"Could not connect to server or stream failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def audio_chunk_generator(stream):
    """
    A generator that yields audio chunks for gRPC.
    Uses VAD to detect speech and silence to know when to stop.
    """
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    
    # Buffer to hold audio chunks before speech is detected
    pre_speech_buffer = collections.deque(maxlen=PRE_SPEECH_BUFFER_CHUNKS)
    
    triggered = False
    silence_chunks = 0

    print("✅ Listening for command...")
    for chunk_np in stream:
        chunk_bytes = (chunk_np * 32767).astype(np.int16).tobytes()

        # If not yet triggered, fill the pre-speech buffer
        if not triggered:
            pre_speech_buffer.append(chunk_bytes)
        
        is_speech = vad.is_speech(chunk_bytes, SAMPLE_RATE)

        if is_speech:
            if not triggered:
                # Start of speech detected, send the buffered audio first
                print("Speech detected, streaming...")
                triggered = True
                for buffered_chunk in pre_speech_buffer:
                    yield audiostream_pb2.AudioChunk(audio_chunk=buffered_chunk)
                pre_speech_buffer.clear()

            # Send the current speech chunk
            yield audiostream_pb2.AudioChunk(audio_chunk=chunk_bytes)
            silence_chunks = 0
        elif triggered:
            # We are in the middle of a command, send some silence for natural pauses
            yield audiostream_pb2.AudioChunk(audio_chunk=chunk_bytes)
            silence_chunks += 1
            if silence_chunks > SILENCE_CHUNKS_TRIGGER:
                print("End of command detected.")
                break # Stop the generator
    
    # If the loop finishes without detecting speech, we still need to exit gracefully
    if not triggered:
        print("No command heard.")


def main():
    """Main loop to listen for wake word and then stream commands."""
    # Setup LiveSpeech for wake word detection
    # Note: Using 'keyphrase' is more efficient than a large dictionary for a single phrase.
    speech = LiveSpeech(
        verbose=False,
        sampling_rate=SAMPLE_RATE,
        buffer_size=CHUNK_SIZE,
        no_search=False,
        full_utt=False,
        keyphrase=WAKE_WORD,
        kws_threshold=1e-20 # Adjust this threshold for sensitivity
    )

    # Main application loop
    while True:
        print(f"✅ ACU is running. Waiting for '{WAKE_WORD}'...")
        for phrase in speech:
            # Wake word detected
            print(f"✅ Wake word detected!")
            
            # Open a new stream for command capture
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=CHUNK_SIZE) as stream:
                # Create the generator for this specific command
                audio_iterator = audio_chunk_generator(stream)
                # Stream the command to the brain
                stream_audio_to_brain(audio_iterator)
            
            # Break to restart the LiveSpeech loop for the next wake word
            break

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.")