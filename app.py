import os
import csv
import json
import time
from typing import Literal
from datetime import datetime
from pathlib import Path

import speech_recognition as sr
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from gtts import gTTS
import pygame
import google.generativeai as genai

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Use latest Gemini model
model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

# FastAPI app
app = FastAPI()
chat_history = []
product_data = []

# ========================
# Auto-load Product Data
# ========================
def load_local_product_data():
    global product_data
    try:
        if os.path.exists("products.json"):
            with open("products.json", "r") as f:
                product_data = json.load(f)
        elif os.path.exists("products.csv"):
            with open("products.csv", "r") as f:
                reader = csv.DictReader(f)
                product_data = [row for row in reader]
    except Exception as e:
        print(f"Error loading product data: {e}")

# ========================
# Upload Product File
# ========================
@app.post("/upload_products")
async def upload_products(file: UploadFile = File(...)):
    global product_data
    try:
        if file.filename.endswith(".csv"):
            content = (await file.read()).decode().splitlines()
            reader = csv.DictReader(content)
            product_data = [row for row in reader]
        elif file.filename.endswith(".json"):
            product_data = json.loads((await file.read()).decode())
        else:
            raise HTTPException(status_code=400, detail="Only CSV or JSON files supported.")
        return {"message": "✅ Product info uploaded."}
    except Exception as e:
        return {"error": str(e)}

# ========================
# Detect Conversation Stage
# ========================
def detect_conversation_stage(prompt, role):
    stage_detection_prompt = (
        f"You are a role-play conversation assistant helping in a {role} scenario.\n"
        f"Based on the user's message below, determine the most likely conversation stage.\n"
        f"Possible stages: opening, discovery, presentation, objection_handling, closing, follow_up, general\n"
        f"User message: \"{prompt}\"\n"
        f"Reply with only one stage keyword (e.g., discovery or presentation)."
    )
    try:
        response = genai.GenerativeModel("models/gemini-1.5-pro-latest").generate_content(stage_detection_prompt)
        detected_stage = response.text.strip().lower()
        valid_stages = {"opening", "discovery", "presentation", "objection_handling", "closing", "follow_up", "general"}
        return detected_stage if detected_stage in valid_stages else "general"
    except Exception:
        return "general"

# ========================
# Generate Chat Response
# ========================
def generate_response(prompt, stage, role):
    if role == "seller" and not product_data:
        load_local_product_data()

    if not stage or stage.lower() == "auto":
        stage = detect_conversation_stage(prompt, role)

    context = {
        "buyer": f"You are a curious buyer. You are currently in the **{stage}** stage of the conversation.",
        "seller": f"You are a helpful and persuasive seller. You are currently in the **{stage}** stage of the conversation."
    }.get(role, f"You are having a product-related discussion (stage: {stage}).")

    products_text = "\n\n".join([
        f"- {p.get('name', 'Unknown')} (${p.get('price', 'N/A')})\n  Description: {p.get('description', '')}\n  Features: {p.get('features', '')}"
        for p in product_data[:10]
    ]) or "No product data available."

    system_prompt = (
        f"ROLE: {role.upper()}\n{context}\n\n=== Product Catalog ===\n{products_text}\n\n"
        f"User says: {prompt}\nRespond naturally and persuasively to continue the role-play conversation."
    )

    try:
        chat_history.append({"role": "user", "parts": [system_prompt]})
        response = model.generate_content(chat_history)
        chat_history.append({"role": "model", "parts": [response.text]})
        return response.text
    except Exception as e:
        return f"❌ Error generating response: {str(e)}"

# ========================
# Speech to Text
# ========================
def speech_to_text(file_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
            return recognizer.recognize_google(audio)
    except Exception:
        return "❌ Speech processing failed."

# ========================
# Text to Speech
# ========================
def text_to_speech(text, filename="response.mp3"):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(filename)
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.quit()
        os.remove(filename)
    except Exception as e:
        print(f"❌ TTS error: {e}")

# ========================
# Stream Text Generator
# ========================
def stream_text(text):
    for line in text.split('\n'):
        yield line + '\n'
        time.sleep(0.2)

# ========================
# Interactive Endpoint
# ========================
@app.post("/interact")
async def interact(
    input_type: Literal["text", "voice"] = Form(...),
    output_type: Literal["text", "voice"] = Form(...),
    role: Literal["seller", "buyer"] = Form(...),
    stage: str = Form("auto"),
    prompt: str = Form(None),
    audio: UploadFile = File(None)
):
    try:
        if input_type == "voice":
            if not audio:
                raise HTTPException(status_code=400, detail="🎤 Audio required.")
            with open("input.wav", "wb") as f:
                f.write(await audio.read())
            prompt = speech_to_text("input.wav")
            if prompt.strip().lower() in ["exit", "quit", "bye"]:
                return JSONResponse(content={"message": "👋 Chat ended by voice command."})
        elif input_type == "text" and not prompt:
            raise HTTPException(status_code=400, detail="Prompt required.")
        elif input_type == "text" and prompt.strip().lower() in ["exit", "quit", "bye"]:
            return JSONResponse(content={"message": "👋 Chat ended by text command."})

        response = generate_response(prompt, stage, role)

        if output_type == "voice":
            text_to_speech(response)
            return JSONResponse(content={"message": "🎧 Voice response ready.", "response": response})

        return StreamingResponse(stream_text(response), media_type="text/plain")

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ========================
# Chat JSON Endpoint
# ========================
class ChatInput(BaseModel):
    prompt: str
    role: Literal["seller", "buyer"]
    stage: str = "auto"

@app.post("/chat")
def chat(data: ChatInput):
    try:
        if not data.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt is required.")
        response = generate_response(data.prompt, data.stage, data.role)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========================
# Optional Root Route
# ========================
@app.get("/")
def root():
    return {"message": "✅ FastAPI Gemini chatbot with dynamic role-play is live!"}
