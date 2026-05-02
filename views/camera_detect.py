import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
import pandas as pd
from fpdf import FPDF

# 1. تحميل الموديل
@st.cache_resource
def load_yolo_model():
    path = 'MODELS/best.pt' if os.path.exists('MODELS/best.pt') else 'best.pt'
    return YOLO(path) if os.path.exists(path) else None

model = load_yolo_model()

# 2. إنشاء التقرير
def create_pdf_report(detections_count, image_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="SafeRoad AI - Official Report", ln=True, align='C')
    pdf.image(image_path, x=10, y=50, w=180)
    report_name = f"Report_{datetime.now().strftime('%H%M%S')}.pdf"
    pdf.output(report_name)
    return report_name

def render_camera_detection():
    st.title("🛰️ SafeRoad AI: Autonomous Detection")
    
    if model is None:
        st.error("Model 'best.pt' missing in MODELS folder!")
        return

    # استخدام الكاميرا المستقرة
    img_file = st.camera_input("SCANNING ROAD")

    if img_file:
        bytes_data = img_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # الرصد التلقائي (conf=0.2 لزيادة الحساسية)
        results = model.predict(cv2_img, conf=0.2)
        num_detections = len(results[0].boxes)
        
        if num_detections > 0:
            # رسم الإطار الأحمر التلقائي
            annotated_img = results[0].plot()
            
            # تنبيهات احترافية
            st.error(f"🚨 ALERT: {num_detections} Potholes Detected!")
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
            
            # حفظ للتقرير
            temp_path = "detected.jpg"
            cv2.imwrite(temp_path, annotated_img)
            report_file = create_pdf_report(num_detections, temp_path)
            
            with open(report_file, "rb") as f:
                st.download_button("📥 Download Official PDF Report", f, file_name=report_file)
        else:
            st.success("🛣️ Road is Clear. Monitoring continues...")

if __name__ == "__main__":
    render_camera_detection()
