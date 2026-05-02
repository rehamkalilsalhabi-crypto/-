import streamlit as st
import cv2
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
from fpdf import FPDF
import tempfile
import os
import time

# استيراد الدوال المساعدة
from utils.alerts import play_alert

def create_live_report_with_images(detections):
    """دالة لإنشاء ملف PDF يحتوي على قائمة الحفر ولقطات الشاشة"""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(200, 20, "SafeRoad AI: Official Maintenance Report", ln=True, align='C')
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
    pdf.ln(10)

    for i, det in enumerate(detections):
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, f"Detection #{i+1}", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, f"Time: {det['Time']} | Confidence: {det['Confidence']}", ln=True)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
            tmpfile.write(det['Image'])
            tmp_path = tmpfile.name
        
        pdf.image(tmp_path, x=10, w=120) # تكبير الصورة قليلاً في التقرير
        pdf.ln(5)
        
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        
        if (i + 1) % 2 == 0:
            pdf.add_page()

    report_path = "Official_Road_Report.pdf"
    pdf.output(report_path)
    return report_path

def render_camera_detection():
    st.markdown("<h1 style='text-align: center; color: #FFD700;'>🛣️ SafeRoad AI: Live Inspection</h1>", unsafe_allow_html=True)
    st.divider()
    
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    MODEL_PATH = PROJECT_ROOT / "MODELS" / "best.pt"
    
    @st.cache_resource
    def load_model():
        return YOLO(str(MODEL_PATH))
    
    model = load_model()

    # إنشاء مخزن للبيانات والتحكم في التكرار
    if "detections_list" not in st.session_state:
        st.session_state.detections_list = []
    if "last_pothole_time" not in st.session_state:
        st.session_state.last_pothole_time = 0

    # إعدادات السرعة والاتصال في القائمة الجانبية
    st.sidebar.subheader("📡 Connection & Performance")
    ip_input = st.sidebar.text_input("Camera IP", value="192.168.0.68:8080")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.65) # رفع العتبة لتقليل الخطأ
    inference_size = st.sidebar.selectbox("Inference Size (Speed)", [320, 640], index=0) # 320 أسرع بكثير
    cooldown = st.sidebar.slider("Detection Cooldown (Seconds)", 1, 10, 3) # وقت الانتظار لمنع التكرار

    camera_url = f"http://{ip_input}/video"
    run = st.checkbox('🔴 Start Live Inspection Scan')
    FRAME_WINDOW = st.image([])

    if run:
        camera = cv2.VideoCapture(camera_url)
        while run:
            ret, frame = camera.read()
            if not ret: 
                st.warning("Failed to connect to camera. Ensure IP Webcam is active.")
                break

            # تحليل الإطارات مع تحديد الحجم لزيادة السرعة
            results = model(frame, conf=conf_threshold, imgsz=inference_size, verbose=False)
            res_plotted = results[0].plot()

            current_time = time.time()
            
            # إذا رصد الموديل حفرة
            if len(results[0].boxes) > 0:
                # التحقق من "وقت الانتظار" لكي لا يسجل نفس الحفرة مراراً
                if current_time - st.session_state.last_pothole_time > cooldown:
                    for box in results[0].boxes:
                        conf = float(box.conf[0])
                        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        _, buffer = cv2.imencode('.jpg', res_plotted)
                        img_bytes = buffer.tobytes()

                        st.session_state.detections_list.append({
                            "Time": timestamp_str,
                            "Confidence": f"{conf:.2%}",
                            "Image": img_bytes
                        })
                        
                        # تحديث وقت آخر رصد وتشغيل التنبيه
                        st.session_state.last_pothole_time = current_time
                        play_alert()

            FRAME_WINDOW.image(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
        camera.release()

    st.divider()
    st.subheader("📊 Detection Evidence Log")
    
    if st.session_state.detections_list:
        # عرض السجلات مرتبة من الأحدث
        for det in reversed(st.session_state.detections_list[-5:]):
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1: st.write(f"🕒 {det['Time']}")
            with col2: st.write(f"🎯 {det['Confidence']}")
            with col3: st.image(det['Image'], width=250)
            st.divider()

        if st.button("📑 Generate Official Report"):
            report_file = create_live_report_with_images(st.session_state.detections_list)
            st.balloons()
            
            with open(report_file, "rb") as f:
                st.download_button(
                    label="📥 Download Official PDF Report",
                    data=f,
                    file_name="SafeRoad_Report.pdf",
                    mime="application/pdf"
                )
    else:
        st.info("No potholes detected yet. The log will appear here.")