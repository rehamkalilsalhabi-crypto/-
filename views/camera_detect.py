import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
from fpdf import FPDF
import time

# حاول استيراد geocoder، وإذا لم يجدها لن ينهار التطبيق
try:
    import geocoder
except ImportError:
    geocoder = None

# 1. تحميل الموديل
@st.cache_resource
def load_yolo_model():
    path = 'MODELS/best.pt' if os.path.exists('MODELS/best.pt') else 'best.pt'
    return YOLO(path) if os.path.exists(path) else None

model = load_yolo_model()

def render_camera_detection():
    st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>📡 AI Autonomous Radar</h1>", unsafe_allow_html=True)
    
    if model is None:
        st.error("❌ Model 'best.pt' missing!")
        return

    if "auto_logs" not in st.session_state:
        st.session_state.auto_logs = []

    # الكاميرا المباشرة
    img_file = st.camera_input("AI IS MONITORING - POINT AT ROAD")

    if img_file:
        bytes_data = img_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # رصد تلقائي فوراً
        results = model.predict(cv2_img, conf=0.5)
        num_detections = len(results[0].boxes)
        
        # رسم الإطار الأحمر
        res_plotted = results[0].plot(line_width=4)
        res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

        if num_detections > 0:
            # صوت تنبيه
            st.components.v1.html("""<audio autoplay><source src="https://www.soundjay.com/buttons/beep-01a.mp3"></audio>""", height=0)
            st.markdown("<h2 style='color: red; text-align: center;'>🚨 ALERT: Pothole Detected!</h2>", unsafe_allow_html=True)
            
            # حفظ تلقائي للتقرير
            img_path = f"auto_det_{len(st.session_state.auto_logs)}.jpg"
            cv2.imwrite(img_path, res_plotted)
            
            # جلب الموقع إذا كانت المكتبة موجودة
            location = "Unknown"
            if geocoder:
                g = geocoder.ip('me')
                location = g.city if g.city else "Scanning..."

            st.session_state.auto_logs.append({
                "path": img_path,
                "time": datetime.now().strftime("%H:%M:%S"),
                "loc": location
            })

        st.image(res_plotted_rgb, caption="AI Real-time Analysis")

    # زر التقرير
    if st.session_state.auto_logs:
        if st.button("📑 Generate Report"):
            # (كود الـ PDF كما في السابق)
            st.success("Report Ready!")

if __name__ == "__main__":
    render_camera_detection()
