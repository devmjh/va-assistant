# Filename: brain_jetson/handler_server.py
# FINAL STABLE VERSION: Includes correct MySQL query and a timeout for the API call.

import os
import sys
import time
import json
import traceback
from dotenv import load_dotenv
from concurrent import futures
import subprocess
import requests
import sounddevice as sd
import numpy as np
import ollama
from openai import OpenAI, Timeout
import mysql.connector

from vosk import Model, KaldiRecognizer
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
HOME_DIR = os.path.expanduser('~')
VOSK_MODEL_PATH = os.path.join(HOME_DIR, 'va-assistant/vosk-model-small-en-us-0.15')
LOCAL_LLM_MODEL = "phi3:mini"
conversation_history = []

# --- Component Initialization ---
client = OpenAI(api_key=api_key, timeout=20.0)
mysql_conn = None
mysql_cursor = None
vosk_model = None

# --- Core AI and Skill Functions ---
def speak(text):
    """
    Sends text to the dedicated TTS server, saves the returned audio,
    and plays it using aplay.
    """
    print(f"TTS: {text}")
    tts_server_url = "http://192.168.4.225:5002/api/tts"
    local_audio_file = "response.wav"

    try:
        payload = {'text': text}
        response = requests.post(tts_server_url, json=payload, timeout=20.0)

        if response.status_code == 200:
            with open(local_audio_file, 'wb') as f:
                f.write(response.content)
            print("--- Playing audio file now... ---")
            subprocess.run(["aplay", local_audio_file], check=True, 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("--- Finished playing audio. ---")
            os.remove(local_audio_file)
        else:
            print(f"Error from TTS Server: {response.status_code} - {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"Could not connect to TTS server: {e}")
    except Exception as e:
        print(f"An error occurred in the speak function: {e}")

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
                query = "SELECT item, quantity FROM lab_inventory"
                mysql_cursor.execute(query)
            else:
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

def add_to_inventory(transcript):
    """Parses a command to add an item to the inventory and updates the database."""
    try:
        words = transcript.lower().split()
        add_index = words.index("add")
        to_index = words.index("to")
        
        num_map = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, 
                   "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
        
        quantity_word = words[add_index + 1]
        quantity = num_map.get(quantity_word, int(quantity_word) if quantity_word.isdigit() else 1)
        
        item_name = " ".join(words[add_index + 2 : to_index])
        
        if not item_name:
            return "I didn't catch the item name. Please try again."

        query = """
            INSERT INTO lab_inventory (item, quantity) 
            VALUES (%s, %s) 
            ON DUPLICATE KEY UPDATE quantity = quantity + VALUES(quantity)
        """
        mysql_cursor.execute(query, (item_name, quantity))
        mysql_conn.commit()
        
        plural = "s" if quantity > 1 else ""
        return f"Okay, I've added {quantity} {item_name}{plural} to the inventory."

    except (ValueError, IndexError):
        return "I didn't understand the format. Please say something like 'add one item to the inventory'."
    except Exception as e:
        print(f"A database error occurred during insert: {e}")
        return "Sorry, I had a problem updating the database."

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
            # --- The full intent parser ---
            if "add" in transcript.lower() and "inventory" in transcript.lower():
                result = add_to_inventory(transcript)
            elif "inventory" in transcript.lower():
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