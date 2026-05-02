import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
import pandas as pd
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

# 2. دالة بناء التقرير PDF
def create_pdf_report(detections_count, image_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="SafeRoad AI - Detection Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Potholes Detected: {detections_count}", ln=True, align='L')
    pdf.image(image_path, x=10, y=50, w=180)
    report_name = "SafeRoad_Official_Report.pdf"
    pdf.output(report_name)
    return report_name

def render_camera_detection():
    st.title("🤖 Autonomous Pothole Monitor")
    st.write("---")

    if model is None:
        st.error("Model 'best.pt' not found. Please check your GitHub folders.")
        return

    st.info("System is in **Auto-Scan Mode**. Point the camera at the road.")

    # استخدام دخل الكاميرا الأساسي
    img_file = st.camera_input("Scanning Environment...", key="auto_scan")

    if img_file:
        # تحويل الصورة ومعالجتها فوراً
        bytes_data = img_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # الرصد بحساسية عالية 0.2
        results = model.predict(cv2_img, conf=0.2, verbose=False)
        num_detections = len(results[0].boxes)
        annotated_img = results[0].plot()

        # إذا لقط حفرة:
        if num_detections > 0:
            st.warning(f"🚨 ALERT: {num_detections} Pothole(s) Auto-Detected!")
            
            # حفظ الصورة للتقرير
            temp_path = "captured_pothole.jpg"
            cv2.imwrite(temp_path, annotated_img)
            
            # عرض النتيجة
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Detection Success")
            
            # توليد التقرير تلقائياً
            report_file = create_pdf_report(num_detections, temp_path)
            
            # زر التحميل يظهر فوراً
            with open(report_file, "rb") as f:
                st.download_button(
                    label="📥 Download This Report Now",
                    data=f,
                    file_name=f"SafeRoad_Report_{datetime.now().strftime('%H%M%S')}.pdf",
                    mime="application/pdf"
                )
                
            # حفظ في سجل البيانات CSV
            if not os.path.exists("detections_log.csv"):
                pd.DataFrame(columns=["Time", "Count"]).to_csv("detections_log.csv", index=False)
            
            new_log = pd.DataFrame({"Time": [datetime.now()], "Count": [num_detections]})
            new_log.to_csv("detections_log.csv", mode='a', header=False, index=False)
            st.toast("Result saved to system logs!", icon="✅")
        else:
            st.success("✅ Road area clear. Continue scanning.")

if __name__ == "__main__":
    render_camera_detection()
