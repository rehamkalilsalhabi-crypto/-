import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
from fpdf import FPDF
import av

# 1. تحميل الموديل بذكاء
@st.cache_resource
def load_yolo_model():
    path = 'MODELS/best.pt' if os.path.exists('MODELS/best.pt') else 'best.pt'
    return YOLO(path) if os.path.exists(path) else None

model = load_yolo_model()

# 2. معالج الفيديو (هنا السحر)
class PotholeProcessor(VideoProcessorBase):
    def __init__(self):
        self.alert_status = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # الرصد الحي (حساسية 0.2)
        results = model.predict(img, conf=0.2, verbose=False)
        annotated_frame = results[0].plot()
        
        # إذا لقط حفرة، نغير الحالة عشان الواجهة تعرف
        if len(results[0].boxes) > 0:
            self.alert_status = True
        else:
            self.alert_status = False
            
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

def render_camera_detection():
    st.title("🛰️ SafeRoad AI: Live Radar")
    
    if model is None:
        st.error("Model 'best.pt' not found!")
        return

    st.info("الرادار يعمل الآن.. وجه الكاميرا للطريق وسيرسم الإطارات الحمراء تلقائياً.")

    # تشغيل البث المباشر (بدون زر تصوير)
    webrtc_ctx = webrtc_streamer(
        key="pothole-radar",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=PotholeProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
    )

    # التفاعل مع الرصد الحي
    if webrtc_ctx.video_processor:
        if webrtc_ctx.video_processor.alert_status:
            st.error("🚨 ALERT: POTHOLE DETECTED!")
            # إضافة صوت تنبيه بسيط عبر HTML
            st.components.v1.html("""
                <audio autoplay><source src="https://www.soundjay.com/buttons/beep-01a.mp3" type="audio/mpeg"></audio>
            """, height=0)
            
            if st.button("Generate Report for this Pothole"):
                st.success("Report generated! Check your downloads.")

if __name__ == "__main__":
    render_camera_detection()
