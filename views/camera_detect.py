import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
import pandas as pd
from fpdf import FPDF
import time

# 1. تحميل الموديل بذكاء من المجلد MODELS
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
    pdf.cell(200, 10, txt="SafeRoad AI - Automated Detection Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Status: Alert - Potholes Detected", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Total Detections: {detections_count}", ln=True, align='L')
    pdf.image(image_path, x=10, y=50, w=180)
    report_name = "Automated_Road_Report.pdf"
    pdf.output(report_name)
    return report_name

def render_camera_detection():
    st.title("🤖 Live Autonomous Detection")
    st.write("---")

    if model is None:
        st.error("Model 'best.pt' not found in MODELS folder.")
        return

    # شرح للمستخدم
    st.info("System is monitoring... Point the camera at the road. It will auto-detect and generate reports.")

    # استخدام ميزة الكاميرا ولكن مع معالجة مستمرة
    # ملاحظة: في الويب، يجب ضغط الزر مرة واحدة للبدء بسبب سياسات الخصوصية
    img_file = st.camera_input("LIVE MONITORING ACTIVE")

    if img_file:
        bytes_data = img_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # الرصد التلقائي بحساسية عالية 0.2 (عشان يلقط بسرعة)
        results = model.predict(cv2_img, conf=0.2, verbose=False)
        num_detections = len(results[0].boxes)
        
        if num_detections > 0:
            # هنا يظهر "الإطار الأحمر" والتحذير فوراً
            annotated_img = results[0].plot()
            
            # تنبيه بصري وسمعي (إشعار الموقع)
            st.warning(f"🚨 ALERT! {num_detections} POTHOLE(S) DETECTED!")
            st.toast("⚠️ Pothole Detected!", icon="🚨")
            
            # عرض النتيجة فوراً
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Live Detection Result")
            
            # حفظ الصورة وتوليد التقرير تلقائياً خلف الكواليس
            temp_path = "auto_capture.jpg"
            cv2.imwrite(temp_path, annotated_img)
            report_file = create_pdf_report(num_detections, temp_path)
            
            # حفظ في السجل CSV
            new_log = pd.DataFrame({"Time": [datetime.now()], "Count": [num_detections]})
            new_log.to_csv("detections_log.csv", mode='a', header=not os.path.exists("detections_log.csv"), index=False)

            # زر التحميل يظهر مباشرة بعد الرصد الآلي
            with open(report_file, "rb") as f:
                st.download_button(
                    label="📥 Download Detection Report",
                    data=f,
                    file_name=f"SafeRoad_Report_{datetime.now().strftime('%H%M%S')}.pdf",
                    mime="application/pdf"
                )
        else:
            st.success("✅ Road is Clear. Monitoring continues...")

if __name__ == "__main__":
    render_camera_detection()
