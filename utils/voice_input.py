# utils/voice_input.py

import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO

def record_and_transcribe(audio_bytes):
    audio = AudioSegment.from_file(BytesIO(audio_bytes), format="webm")
    wav_io = BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)

    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_io) as source:
        audio_data = recognizer.record(source)

    try:
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return "❌ Could not understand audio."
    except sr.RequestError:
        return "❌ Speech recognition service error."
