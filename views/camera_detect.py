import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
from fpdf import FPDF
import time

# 1. تحميل الموديل بذكاء
@st.cache_resource
def load_yolo_model():
    search_paths = ['MODELS/best.pt', 'models/best.pt', 'best.pt']
    for path in search_paths:
        if os.path.exists(path):
            return YOLO(path)
    return None

model = load_yolo_model()

# 2. دالة توليد التقرير التلقائي
def generate_auto_report(detections_count, image_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="SafeRoad AI - Autonomous Detection Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Potholes Identified: {detections_count}", ln=True, align='L')
    pdf.image(image_path, x=10, y=50, w=180)
    
    report_path = "Official_SafeRoad_Report.pdf"
    pdf.output(report_path)
    return report_path

def render_camera_detection():
    st.title("🤖 Autonomous Road Monitoring")
    st.markdown("---")

    if model is None:
        st.error("Model not found! Check MODELS folder.")
        return

    # التنبيه الصوتي (سيتم تشغيله عند الرصد)
    def play_alert():
        st.markdown("""
            <audio autoplay>
                <source src="https://www.soundjay.com/buttons/beep-01a.mp3" type="audio/mpeg">
            </audio>
        """, unsafe_allow_content=True)

    # حاوية العرض المباشر
    status_placeholder = st.empty()
    image_placeholder = st.empty()

    # فتح الكاميرا (ملاحظة: المتصفح يتطلب ضغط الزر مرة واحدة للأمان)
    img_file = st.camera_input("SCANNING ACTIVE", key="monitoring_active")

    if img_file:
        bytes_data = img_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # الرصد التلقائي بحساسية عالية جداً لسرعة الاستجابة
        results = model.predict(cv2_img, conf=0.15, verbose=False)
        num_detections = len(results[0].boxes)
        
        if num_detections > 0:
            # رسم الإطار الأحمر التلقائي
            annotated_img = results[0].plot()
            
            # تشغيل صوت التنبيه
            play_alert()
            
            # عرض النتيجة مع "إطار أحمر" افتراضي في الواجهة
            st.error(f"🚨 ALERT: {num_detections} Pothole(s) Detected!")
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Snapshot Captured!")

            # حفظ الصورة وتوليد التقرير خلف الكواليس
            temp_path = "auto_detect.jpg"
            cv2.imwrite(temp_path, annotated_img)
            report_file = generate_auto_report(num_detections, temp_path)

            # توفير التقرير للتحميل فوراً
            with open(report_file, "rb") as f:
                st.download_button(
                    label="📥 Download Official Report (PDF)",
                    data=f,
                    file_name=f"SafeRoad_Report_{datetime.now().strftime('%H%M%S')}.pdf",
                    mime="application/pdf"
                )
            
            st.toast("Pothole logged to system logs!", icon="🚨")
        else:
            st.success("🛣️ Road is Clear. Monitoring continues...")

if __name__ == "__main__":
    render_camera_detection()
