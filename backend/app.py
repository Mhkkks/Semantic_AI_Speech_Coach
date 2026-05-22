from fastapi import FastAPI
from fastapi import UploadFile, File, Form

from pipeline import analyze_speech
from feedback import generate_feedback

import shutil

app = FastAPI()

@app.post("/analyze")

async def analyze(

    transcript: str = Form(...),

    audio: UploadFile = File(...)

):

    audio_path = f"uploads/{audio.filename}"

    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    results = analyze_speech(
        transcript,
        audio_path
    )

    feedback = generate_feedback(results)

    results["feedback"] = feedback

    return results