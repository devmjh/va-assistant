# filename: mic_test.py
import sounddevice as sd
import numpy as np
import time

SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
DURATION = 10  # Run for 10 seconds

print("--- Starting Microphone Test ---")
print(f"Listening for {DURATION} seconds...")

try:
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', blocksize=CHUNK_SIZE) as stream:
        for i in range(int(DURATION * SAMPLE_RATE / CHUNK_SIZE)):
            chunk, overflowed = stream.read(CHUNK_SIZE)
            
            # Calculate the volume (Root Mean Square)
            rms = np.sqrt(np.mean(chunk.astype(np.float32)**2))
            
            # Create a simple visual volume meter
            volume_bar = '#' * int(rms / 100)
            print(f"Volume: {volume_bar}")
            
            time.sleep(0.05) # Slow down the printing a little

except Exception as e:
    print(f"An error occurred: {e}")

print("--- Test Finished ---")