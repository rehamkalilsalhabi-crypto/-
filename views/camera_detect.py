import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# تحميل الموديل
@st.cache_resource
def load_model():
    path = 'MODELS/best.pt' if os.path.exists('MODELS/best.pt') else 'best.pt'
    if os.path.exists(path):
        return YOLO(path)
    return None

model = load_model()

st.title("🛡️ SafeRoad AI - LIVE Detection")

if model is None:
    st.error("Model not found!")
    st.stop()

st.warning("🚨 SYSTEM ACTIVE - LIVE SCANNING")

# كلاس لمعالجة الفيديو مباشرة
class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = model.predict(img, conf=0.25, verbose=False)

        if len(results[0].boxes) > 0:
            img = results[0].plot()

        return img

# تشغيل الكاميرا مباشرة
webrtc_streamer(
    key="live-detection",
    video_processor_factory=VideoProcessor
)
