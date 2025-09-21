# AI Voice Assistant for the Home Lab 🤖

## Overview

This project creates a distributed, on-device AI voice assistant designed for a home technology lab. It uses a modular, client-server architecture to separate audio capture from AI processing, resulting in a robust and scalable system.

The assistant is triggered by a wake word, transcribes spoken commands, parses user intent, and interfaces with local AI models, databases, and cloud APIs to provide intelligent responses.

---

## Architecture

The system is broken into two primary components that communicate over the local network using a custom gRPC audio stream.

### 1. The Audio Capture Unit (ACU) 👂
* **Hardware:** A Raspberry Pi 5 with a connected USB microphone.
* **Software:** A dedicated Python script (`listener_client.py`).
* **Role:** The ACU's sole purpose is to listen for the wake word using the lightweight PocketSphinx engine. When detected, it records the user's command and streams the audio data in real-time to the AI Brain.

### 2. The AI Brain 🧠
* **Hardware:** An NVIDIA Jetson Orin Nano with connected speakers.
* **Software:** The main Python server script (`handler_server.py`).
* **Role:** The Brain acts as a gRPC server, waiting for an audio stream from any ACU on the network. When it receives audio, it performs all heavy AI processing:
    * **ASR:** Transcribes speech to text using the **Vosk** offline model.
    * **Intent Parsing:** Determines if the command is a local skill or a general query.
    * **Skills:** Executes local functions, such as querying a **MySQL** database or checking system status.
    * **LLM Integration:** Forwards queries to a local **Ollama** model or a remote **OpenAI API**.
    * **TTS:** Speaks the final response using the **pyttsx3** offline engine.

---

## Project Structure

va-assistant/
├── .gitignore
├── README.md
├── protos/
│   └── audiostream.proto
├── acu_pi/
│   └── listener_client.py
└── brain_jetson/
└── handler_server.py

---

## Core Technologies

* **Wake Word:** PocketSphinx
* **ASR (Offline):** Vosk
* **TTS (Offline):** pyttsx3
* **Local LLM:** Ollama
* **Cloud LLM:** OpenAI API
* **Networking:** gRPC
* **Database:** MySQL
* **Hardware:** NVIDIA Jetson Orin Nano, Raspberry Pi 5

---

## Usage

The system requires starting three processes in separate terminals:

1.  **AI Brain (on the Jetson):** Start the `handler_server.py` script. This will initialize the AI models and the gRPC server.
2.  **ACU (on the Raspberry Pi):** Start the `listener_client.py` script. This will begin listening for the wake word.
3.  *(Optional) Other Services:* Start any other required services, like the Ollama server.

