# filename: tts_app.py
import os
from flask import Flask, request, send_file, jsonify
from TTS.api import TTS
import torch
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)

# --- Server Setup ---
app = Flask(__name__)
PORT = 5002
OUTPUT_FILENAME = "output.wav"

# --- TTS Model Initialization ---
# Check if CUDA (GPU) is available and select it
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"TTS running on device: {device}")

# Load a high-quality TTS model
# You can find more models here: https://huggingface.co/coqui/XTTS-v2
# Or by running `tts --list_models` in your terminal
model_name = "tts_models/en/ljspeech/tacotron2-DDC"
logging.info(f"Loading TTS model: {model_name}...")
tts = TTS(model_name=model_name, progress_bar=False).to(device)
logging.info("TTS model loaded successfully.")


# --- API Endpoint ---
@app.route('/api/tts', methods=['POST'])
def generate_speech():
    """
    Receives text in a JSON payload and returns a WAV audio file.
    """
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    text_to_speak = data['text']
    if not text_to_speak.strip():
        return jsonify({"error": "Text cannot be empty"}), 400

    logging.info(f"Received request to synthesize: '{text_to_speak}'")

    try:
        # Generate speech and save to a file
        tts.tts_to_file(text=text_to_speak, file_path=OUTPUT_FILENAME)

        # Check if the file was created
        if not os.path.exists(OUTPUT_FILENAME):
            return jsonify({"error": "TTS failed to generate audio file"}), 500
        
        logging.info(f"Successfully generated audio file: {OUTPUT_FILENAME}")

        # Send the audio file back to the client
        return send_file(
            OUTPUT_FILENAME,
            mimetype="audio/wav",
            as_attachment=True,
            download_name="response.wav"
        )
    except Exception as e:
        logging.error(f"An error occurred during TTS synthesis: {e}")
        return jsonify({"error": "An internal server error occurred"}), 500

# --- Main Execution ---
if __name__ == '__main__':
    # Setting host='0.0.0.0' makes it accessible on your local network
    app.run(host='0.0.0.0', port=PORT, debug=True)