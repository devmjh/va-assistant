# filename: debug_tts.py
from TTS.api import TTS
import torch

print("Loading model...")
model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    tts = TTS(model_name=model_name).to(device)
    print("Model loaded successfully.")

    print("\n--- Inspecting the main 'tts' object ---")
    print(f"Type: {type(tts)}")
    print(f"Attributes: {dir(tts)}")

    print("\n--- Inspecting the 'tts.synthesizer' object ---")
    if hasattr(tts, 'synthesizer'):
        print(f"Type: {type(tts.synthesizer)}")
        print(f"Attributes: {dir(tts.synthesizer)}")

        print("\n--- Inspecting the 'tts.synthesizer.tts_model' object ---")
        if hasattr(tts.synthesizer, 'tts_model'):
            print(f"Type: {type(tts.synthesizer.tts_model)}")
            print(f"Attributes: {dir(tts.synthesizer.tts_model)}")
        else:
            print("'tts.synthesizer' has no 'tts_model' attribute.")
    else:
        print("'tts' object has no 'synthesizer' attribute.")

except Exception as e:
    print(f"An error occurred: {e}")