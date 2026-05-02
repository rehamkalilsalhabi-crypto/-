import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
from fpdf import FPDF

# 1. تحميل الموديل
@st.cache_resource
def load_yolo_model():
    # يبحث في المجلد الذي ظهر في صورتك MODELS
    path = 'MODELS/best.pt' if os.path.exists('MODELS/best.pt') else 'best.pt'
    if os.path.exists(path):
        return YOLO(path)
    return None

model = load_yolo_model()

# 2. دالة التنبيه الصوتي
def trigger_alert():
    # كود جافا سكريبت لتشغيل صوت تنبيه
    st.components.v1.html("""
        <audio autoplay><source src="https://www.soundjay.com/buttons/beep-01a.mp3" type="audio/mpeg"></audio>
    """, height=0)

def render_camera_detection():
    st.title("🛡️ SafeRoad AI: Autonomous Monitor")
    
    if model is None:
        st.error("Model 'best.pt' missing in MODELS folder!")
        return

    st.warning("SYSTEM STATUS: LIVE SCANNING...")

    # هذا المكون سيظهر الكاميرا، وبمجرد التقاط الصورة سيعالجها الموديل فوراً
    # ويرسم الإطار الأحمر "تلقائياً"
    img_file = st.camera_input("POINT AT ROAD - AUTO DETECTION ACTIVE")

    if img_file:
        bytes_data = img_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # الرصد بحساسية عالية جداً (0.15) لضمان عدم فوات أي حفرة
        results = model.predict(cv2_img, conf=0.15, verbose=False)
        num_detections = len(results[0].boxes)
        
        if num_detections > 0:
            # هنا يظهر "الإطار الأحمر" الذي يحدد الحفرة
            annotated_img = results[0].plot()
            
            # تشغيل صوت التنبيه
            trigger_alert()
            
            # عرض النتيجة مع الإطار الأحمر فوراً
            st.error(f"🚨 POTHOLE DETECTED! (Count: {num_detections})")
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)

            # توليد التقرير PDF آلياً خلف الكواليس
            temp_path = "detected_pothole.jpg"
            cv2.imwrite(temp_path, annotated_img)
            
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, txt="Official Detection Report", ln=True, align='C')
            pdf.image(temp_path, x=10, y=40, w=180)
            pdf.output("Report.pdf")

            # زر التحميل يظهر جاهزاً
            with open("Report.pdf", "rb") as f:
                st.download_button("📥 Download Report", f, "SafeRoad_Report.pdf")
        else:
            st.success("🛣️ Road is Clear.")

if __name__ == "__main__":
    render_camera_detection()
