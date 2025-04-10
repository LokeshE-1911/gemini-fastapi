import os
import csv
import json
from typing import Literal

import speech_recognition as sr
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from gtts import gTTS
from playsound import playsound

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

app = FastAPI()
chat_history = []
product_data = {}

# ------------------- Load Product File -------------------
@app.post("/upload_products")
async def upload_products(file: UploadFile = File(...)):
    try:
        global product_data
        if file.filename.endswith(".csv"):
            reader = csv.DictReader((await file.read()).decode().splitlines())
            product_data = [row for row in reader]
        elif file.filename.endswith(".json"):
            product_data = json.loads((await file.read()).decode())
        else:
            raise HTTPException(status_code=400, detail="Only CSV or JSON files supported.")
        return {"message": "Product information uploaded successfully."}
    except Exception as e:
        return {"error": str(e)}

# ------------------- Utilities -------------------
def speech_to_text(file_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
            return recognizer.recognize_google(audio_data)
    except Exception:
        return "Speech processing failed."

def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        filename = "response.mp3"
        tts.save(filename)
        playsound(filename)
        os.remove(filename)
    except Exception as e:
        print(f"❌ Text-to-speech error: {e}")

# ------------------- Prompt Generation -------------------
def generate_response(prompt, stage, role):
    context = {
        "buyer": "You are a potential customer exploring the products listed. Ask relevant questions, compare features, and show interest.",
        "seller": "You are a product sales rep. Promote the products effectively, highlight benefits, answer questions persuasively."
    }.get(role, "You are helping with product-related queries.")

    products_text = "\n".join([f"- {item['name']} (${item['price']})" for item in product_data]) if product_data else "No product data available."

    system_prompt = (
        f"{context}\n\nAvailable Products:\n{products_text}\n\nConversation Context: {stage}.\nCustomer/User: {prompt}"
    )

    try:
        chat_history.append({"role": "user", "parts": [system_prompt]})
        response = model.generate_content(chat_history)
        chat_history.append({"role": "model", "parts": [response.text]})
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

# ------------------- Endpoints -------------------
@app.post("/interact")
async def interact(
    input_type: Literal["text", "voice"] = Form(...),
    output_type: Literal["text", "voice"] = Form(...),
    role: Literal["seller", "buyer"] = Form(...),
    stage: str = Form("general"),
    prompt: str = Form(None),
    audio: UploadFile = File(None)
):
    try:
        if input_type == "voice":
            if not audio:
                raise HTTPException(status_code=400, detail="No audio file provided.")
            audio_path = "input.wav"
            with open(audio_path, "wb") as f:
                f.write(await audio.read())
            prompt = speech_to_text(audio_path)
        elif input_type == "text" and not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required.")

        response = generate_response(prompt, stage, role)

        if output_type == "voice":
            text_to_speech(response)
            return JSONResponse(content={"message": "Voice response completed.", "response": response})
        return {"response": response}

    except Exception as e:
        return {"error": str(e)}

class ChatInput(BaseModel):
    prompt: str
    role: Literal["seller", "buyer"]
    stage: str = "general"

@app.post("/chat")
def chat(data: ChatInput):
    try:
        if not data.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt is required.")
        response = generate_response(data.prompt, data.stage, data.role)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
