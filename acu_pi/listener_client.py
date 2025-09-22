# Filename: listener_client.py
# Runs on the Raspberry Pi (ACU) to capture and stream audio.

import os
import sys
import time
import grpc
import sounddevice as sd
import numpy as np
from pocketsphinx import Decoder
import traceback

# --- Add protos directory to path ---
# This allows us to import our generated gRPC modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'protos'))

import audiostream_pb2
import audiostream_pb2_grpc

# --- Configuration ---
# IMPORTANT: Change this to the actual IP address of your Jetson Nano (the "Brain")
SERVER_ADDRESS = 'rdunano2.local:50051' 

# Audio Config
KWS_SAMPLE_RATE = 16000
INPUT_DEVICE = 'Blue Snowball' # Or your ReSpeaker, once it arrives and is tested
KEYPHRASE = "bridge to engineering"
COMMAND_RECORD_SECONDS = 5
CHUNK_SIZE = 1024

# Path Config
HOME_DIR = os.path.expanduser('~')
POCKETSPHINX_MODEL_PATH = os.path.join(HOME_DIR, 'cmusphinx-en-us-ptm-5.2')

# --- gRPC Streaming Function ---
def stream_audio_to_server(audio_data):
    print(f"Connecting to server at {SERVER_ADDRESS}...")
    try:
        with grpc.insecure_channel(SERVER_ADDRESS) as channel:
            stub = audiostream_pb2_grpc.AudioStreamerStub(channel)
            
            # Create a generator to stream audio chunks
            def chunk_generator(data, chunk_size):
                for i in range(0, len(data), chunk_size):
                    yield audiostream_pb2.Chunk(audio_chunk=data[i:i+chunk_size])

            # Send the stream and get the server's response
            response = stub.StreamAudio(chunk_generator(audio_data, CHUNK_SIZE * 2))
            print(f"Server response: '{response.status_message}'")

    except grpc.RpcError as e:
        print(f"Could not connect to server: {e.details()}")
    except Exception as e:
        print(f"An error occurred during streaming: {e}")

# --- Main Application Loop ---
def main():
    print("Initializing PocketSphinx for wake word detection...")
    config = Decoder.default_config()
    config.set_string('-hmm', POCKETSPHINX_MODEL_PATH)
    config.set_string('-keyphrase', KEYPHRASE)
    config.set_float('-kws_threshold', 1e-20)
    config.set_string('-lm', None) 
    decoder = Decoder(config)

    # Outer loop to be resilient to errors
    while True:
        stream = None
        try:
            stream = sd.InputStream(device=INPUT_DEVICE, samplerate=KWS_SAMPLE_RATE, channels=1, dtype='int16', blocksize=CHUNK_SIZE)
            
            print(f"\n✅ ACU is running. Waiting for '{KEYPHRASE}'...")
            stream.start()
            decoder.start_utt()
            
            command_frames = []
            
            # Inner loop for active listening
            while True:
                buf, overflowed = stream.read(CHUNK_SIZE)
                decoder.process_raw(buf.ravel().tobytes(), False, False)
                
                # If wake word is detected, start capturing the command audio
                if decoder.hyp() is not None:
                    print(f"✅ Wake word detected! Recording command for {COMMAND_RECORD_SECONDS} seconds...")
                    
                    # Record the command audio for the specified duration
                    num_chunks_for_command = int((COMMAND_RECORD_SECONDS * KWS_SAMPLE_RATE) / CHUNK_SIZE)
                    for _ in range(num_chunks_for_command):
                        buf, overflowed = stream.read(CHUNK_SIZE)
                        command_frames.append(buf)
                    
                    # Command is recorded, break the inner loop to process it
                    break
            
            # Stop and close the stream to be clean
            stream.stop()
            stream.close()
            
            # Concatenate all the audio frames and stream them to the server
            command_audio = np.concatenate(command_frames)
            stream_audio_to_server(command_audio.tobytes())
            
            # Reset the decoder for the next wake word
            decoder.end_utt()

        except KeyboardInterrupt:
            print("\nStopping ACU.")
            break # Exit the main while loop
        except Exception as e:
            print(f"An error occurred in the listener loop:")
            traceback.print_exc()
            time.sleep(5)
        finally:
            if stream and not stream.closed:
                stream.close()

if __name__ == '__main__':
    main()
