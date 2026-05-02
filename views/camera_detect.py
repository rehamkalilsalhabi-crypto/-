import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import cv2
from ultralytics import YOLO
import os

# تحميل الموديل
@st.cache_resource
def load_yolo_model():
    path = 'MODELS/best.pt' if os.path.exists('MODELS/best.pt') else 'best.pt'
    return YOLO(path) if os.path.exists(path) else None

model = load_yolo_model()

class PotholeProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model.predict(img, conf=0.2, verbose=False)
        annotated_frame = results[0].plot()
        return frame.from_ndarray(annotated_frame, format="bgr24")

def render_camera_detection():
    st.title("🤖 SafeRoad AI: Live Radar")
    if model:
        webrtc_streamer(
            key="pothole-radar",
            video_processor_factory=PotholeProcessor,
            media_stream_constraints={"video": True, "audio": False},
        )

if __name__ == "__main__":
    render_camera_detection()
