import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
from fpdf import FPDF
import time

# 1. تحميل الموديل
@st.cache_resource
def load_yolo_model():
    path = 'MODELS/best.pt' if os.path.exists('MODELS/best.pt') else 'best.pt'
    return YOLO(path) if os.path.exists(path) else None

model = load_yolo_model()

# 2. دالة التقرير (تجمع صور الحفر فقط)
def create_pothole_report(detections):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(200, 20, txt="SafeRoad AI - Potholes Evidence Report", ln=True, align='C')
    pdf.ln(10)

    for i, det in enumerate(detections):
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, txt=f"Pothole Detection #{i+1}", ln=True)
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 10, txt=f"Time: {det['time']} | Confidence: {det['conf']}", ln=True)
        pdf.image(det['path'], x=10, w=150)
        pdf.ln(10)
        if (i + 1) % 2 == 0: pdf.add_page()

    report_path = "Potholes_Only_Report.pdf"
    pdf.output(report_path)
    return report_path

def render_camera_detection():
    st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>📡 Auto-Scanning Radar (5s)</h1>", unsafe_allow_html=True)
    
    # مخزن الصور التي تحتوي على حفر فقط
    if "pothole_logs" not in st.session_state:
        st.session_state.pothole_logs = []

    # إعداد التحديث التلقائي كل 5 ثوانٍ (الزر سيضغط نفسه)
    st.info("⏱️ النظام يقوم بالمسح تلقائياً كل 5 ثوانٍ...")
    
    # مكون الكاميرا
    img_file = st.camera_input("POINT CAMERA AT ROAD")

    if img_file:
        bytes_data = img_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # تحليل الصورة
        results = model.predict(cv2_img, conf=0.5)
        num_detections = len(results[0].boxes)
        
        # رسم الإطار الأحمر
        res_plotted = results[0].plot(line_width=4)
        res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

        # --- المنطق الذكي للتقرير ---
        if num_detections > 0:
            # تشغيل الصوت
            st.components.v1.html("""<audio autoplay><source src="https://www.soundjay.com/buttons/beep-01a.mp3"></audio>""", height=0)
            st.error(f"🚨 ALERT: {num_detections} Pothole detected!")
            
            # حفظ الصورة فقط إذا وجد حفرة
            img_path = f"pothole_{len(st.session_state.pothole_logs)}.jpg"
            cv2.imwrite(img_path, res_plotted)
            
            st.session_state.pothole_logs.append({
                "path": img_path,
                "time": datetime.now().strftime("%H:%M:%S"),
                "conf": f"{float(results[0].boxes[0].conf[0]):.2%}"
            })
        else:
            st.success("🛣️ Road is clear. No potholes to report.")

        st.image(res_plotted_rgb, caption="AI Analysis")

    # زر التقرير النهائي (يجمع صور الحفر فقط)
    if st.session_state.pothole_logs:
        st.divider()
        st.subheader(f"📊 Collected Evidence: {len(st.session_state.pothole_logs)} Potholes")
        if st.button("📑 Generate Potholes-Only Report"):
            report = create_pothole_report(st.session_state.pothole_logs)
            with open(report, "rb") as f:
                st.download_button("📥 Download PDF", f, file_name=report)

    # حيلة التحديث التلقائي: تجعل الصفحة تعيد تحميل نفسها كل 5 ثوانٍ
    time.sleep(5)
    st.rerun()

if __name__ == "__main__":
    render_camera_detection()
