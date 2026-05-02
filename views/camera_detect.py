import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import cv2
import numpy as np
from ultralytics import YOLO

# تحميل الموديل - تأكدي أن المسار 'models/best.pt' موجود فعلاً في GitHub
@st.cache_resource
def load_yolo_model():
    return YOLO('models/best.pt')

model = load_yolo_model()

class PotholeProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # معالجة الإطار باستخدام YOLO (الذكاء الاصطناعي)
        results = model.predict(img, conf=0.5)
        
        # رسم المربعات على الصورة
        annotated_img = results[0].plot()
        
        return frame.from_ndarray(annotated_img, format="bgr24")

def render_camera_detection():
    st.title("🎥 رصد الحفر المباشر (SafeRoad AI)")
    st.write("---")
    
    st.info("💡 اضغط على زر **Start** بالأسفل، ثم وافق على إذن فتح الكاميرا.")

    # هذا الجزء هو البديل الرسمي لكاميرا الـ IP
    webrtc_streamer(
        key="road-live",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=PotholeProcessor,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={
            "video": {
                "facingMode": "environment" # يفتح الكاميرا الخلفية تلقائياً
            },
            "audio": False
        },
        async_processing=True,
    )

if __name__ == "__main__":
    render_camera_detection()
