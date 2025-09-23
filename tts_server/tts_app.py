# filename: tts_server/tts_app.py
import os
import traceback
from flask import Flask, request, send_file, jsonify
from TTS.api import TTS
import torch
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
PORT = 5002
OUTPUT_FILENAME = "output.wav"

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"TTS running on device: {device}")

model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
logging.info(f"Loading TTS model: {model_name}...")
tts = TTS(model_name=model_name, progress_bar=False).to(device)
logging.info("TTS model loaded successfully.")

@app.route('/api/tts', methods=['POST'])
def generate_speech():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    text_to_speak = data['text']
    logging.info(f"Received request to synthesize: '{text_to_speak}'")

    try:
        if os.path.exists(OUTPUT_FILENAME):
            os.remove(OUTPUT_FILENAME)

        # Generate speech using XTTS v2
        tts.tts_to_file(
            text=text_to_speak,
            file_path=OUTPUT_FILENAME,
            language='en'
        )

        if not os.path.exists(OUTPUT_FILENAME):
            return jsonify({"error": "TTS failed to generate audio file"}), 500
        
        return send_file(
            OUTPUT_FILENAME,
            mimetype="audio/wav",
            as_attachment=True,
            download_name="response.wav"
        )
    except Exception as e:
        logging.error(f"An error occurred during TTS synthesis: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=False)