# Filename: handler_server.py
# FINAL STABLE VERSION: Includes correct MySQL query and a timeout for the API call.

import os
import sys
import time
import json
import traceback
from dotenv import load_dotenv
from concurrent import futures
import subprocess

import sounddevice as sd
import numpy as np
import ollama
from openai import OpenAI, Timeout
import mysql.connector

from vosk import Model, KaldiRecognizer
import pyttsx3
import grpc

# --- Add protos directory to path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'protos'))

import audiostream_pb2
import audiostream_pb2_grpc

# --- Configuration ---
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, '.env'))
api_key = os.getenv("OPENAI_API_KEY")
mysql_host = os.getenv("MYSQL_HOST")
mysql_user = os.getenv("MYSQL_USER")
mysql_password = os.getenv("MYSQL_PASSWORD")
mysql_db = os.getenv("MYSQL_DB")

if not api_key: raise ValueError("OPENAI_API_KEY not found in .env")
if not all([mysql_host, mysql_user, mysql_password, mysql_db]): raise ValueError("MySQL credentials not found in .env")

SAMPLE_RATE = 16000
OUTPUT_DEVICE = 'C-Media USB Audio Device'
HOME_DIR = os.path.expanduser('~')
VOSK_MODEL_PATH = os.path.join(HOME_DIR, 'va-assistant/vosk-model-small-en-us-0.15')
LOCAL_LLM_MODEL = "phi3:mini"
conversation_history = []

# --- Component Initialization ---
# Set a timeout for the OpenAI client
client = OpenAI(api_key=api_key, timeout=20.0)
mysql_conn = None
mysql_cursor = None
tts_engine = pyttsx3.init()
vosk_model = None

# --- Core AI and Skill Functions ---
def speak(text):
    print(f"TTS: {text}")
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        print(f"An error occurred during TTS: {e}")

def transcribe_audio_bytes(command_bytes):
    print("Processing command...")
    try:
        rec = KaldiRecognizer(vosk_model, SAMPLE_RATE)
        rec.AcceptWaveform(command_bytes)
        result = json.loads(rec.FinalResult())
        return result.get('text', '')
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        traceback.print_exc()
    return ""

def get_cpu_temperature():
    try:
        temp_str = subprocess.check_output(['cat', '/sys/class/thermal/thermal_zone0/temp']).decode('utf-8')
        temp_c = int(temp_str) / 1000.0
        return f"The current CPU temperature is {temp_c:.1f} degrees Celsius."
    except Exception as e:
        print(f"Could not read CPU temp: {e}")
        return "I was unable to read the CPU temperature."

def local_query(transcript):
    global conversation_history
    if len(conversation_history) > 6: conversation_history = conversation_history[-6:]
    conversation_history.append({"role": "user", "content": transcript})
    response = ollama.chat(model=LOCAL_LLM_MODEL, messages=conversation_history)
    answer = response['message']['content']
    conversation_history.append({"role": "assistant", "content": answer})
    return answer

def local_data_query(transcript):
    if "inventory" in transcript.lower():
        parts = transcript.lower().split("inventory for ")
        item_name = parts[1].strip() if len(parts) > 1 else "all"
        try:
            if item_name == "all":
                # Use the correct column name 'item'
                query = "SELECT item, quantity FROM lab_inventory"
                mysql_cursor.execute(query)
            else:
                # Use the correct column name 'item'
                query = "SELECT item, quantity FROM lab_inventory WHERE item LIKE %s"
                mysql_cursor.execute(query, (f"%{item_name}%",))
            results = mysql_cursor.fetchall()
            mysql_conn.commit()
            if results:
                response = ", ".join([f"{row[1]} {row[0]}" for row in results])
                return f"Current inventory shows: {response}."
            else:
                return f"No inventory found for {item_name}."
        except Exception as e:
            print(f"A database error occurred: {e}")
            return "Sorry, I had a problem querying the database."
    return "No local data query matched."

def api_query(transcript):
    try:
        print("Querying OpenAI API...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": transcript}]
        )
        return response.choices[0].message.content
    except Timeout:
        print("OpenAI API request timed out.")
        return "Sorry, the cloud is not responding quickly enough."
    except Exception as e:
        print(f"An error occurred with the OpenAI API: {e}")
        return "Sorry, I'm having trouble connecting to the cloud at the moment."

# --- gRPC Server Implementation ---
class AudioStreamerServicer(audiostream_pb2_grpc.AudioStreamerServicer):
    def StreamAudio(self, request_iterator, context):
        print("\nConnection received from an ACU...")
        audio_data = bytearray()
        for chunk in request_iterator:
            audio_data.extend(chunk.audio_chunk)
        print(f"Received {len(audio_data)} bytes of audio data.")
        
        speak("Acknowledged.")
        
        transcript = transcribe_audio_bytes(bytes(audio_data))
        
        if transcript:
            print(f"Heard command: '{transcript}'")
            # The full intent parser
            if "inventory" in transcript.lower():
                result = local_data_query(transcript)
            elif "temperature" in transcript.lower():
                result = get_cpu_temperature()
            elif any(kw in transcript.lower() for kw in ["who are you", "what can you do"]):
                result = local_query(transcript)
            else:
                result = api_query(transcript)
            
            print(f"Response: {result}")
            speak(result)
        else:
            speak("I didn't catch that.")
        
        return audiostream_pb2.StreamReceipt(status_message="Audio processed successfully.")

# --- Main Server Function ---
def serve():
    global mysql_conn, mysql_cursor, vosk_model
    try:
        mysql_conn = mysql.connector.connect(
            host=mysql_host, user=mysql_user, password=mysql_password, database=mysql_db
        )
        mysql_cursor = mysql_conn.cursor()
        print("MySQL connection successful.")

        print("Loading Vosk ASR model...")
        if not os.path.exists(VOSK_MODEL_PATH):
            raise FileNotFoundError(f"Vosk model not found at {VOSK_MODEL_PATH}")
        vosk_model = Model(VOSK_MODEL_PATH)
        print("Vosk model loaded.")

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        audiostream_pb2_grpc.add_AudioStreamerServicer_to_server(AudioStreamerServicer(), server)
        server.add_insecure_port('[::]:50051')
        server.start()
        print("✅ AI Brain is running. Listening for audio streams on port 50051...")
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\nStopping AI Brain.")
    except Exception as e:
        print(f"An unexpected error occurred:")
        traceback.print_exc()
    finally:
        if mysql_conn and mysql_conn.is_connected():
            mysql_cursor.close()
            mysql_conn.close()
            print("MySQL connection closed.")

if __name__ == '__main__':
    serve()