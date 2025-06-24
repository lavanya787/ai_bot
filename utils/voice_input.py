import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import speech_recognition as sr
import av
import numpy as np

# --- Audio Processor ---
class AudioProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def recv(self, frame: av.AudioFrame):
        audio = frame.to_ndarray()
        pcm = np.mean(audio, axis=0).astype(np.int16).tobytes()
        try:
            with sr.AudioData(pcm, frame.sample_rate, 2) as audio_data:
                text = self.recognizer.recognize_google(audio_data)
                st.session_state["last_transcript"] = text
        except sr.UnknownValueError:
            st.session_state["last_transcript"] = "Could not understand"
        except sr.RequestError as e:
            st.session_state["last_transcript"] = f"API Error: {e}"

# --- WebRTC UI + Trigger ---

def record_voice():
    if "last_transcript" not in st.session_state:
        st.session_state["last_transcript"] = ""

    st.subheader("🎙️ Voice Input")

    webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        in_audio=True,
        audio_receiver_factory=AudioProcessor
    )

    if st.session_state.get("last_transcript"):
        st.success(f"Transcript: {st.session_state['last_transcript']}")
        return st.session_state["last_transcript"]

    return None
