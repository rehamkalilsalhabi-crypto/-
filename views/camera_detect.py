import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
from fpdf import FPDF

# تحميل الموديل
@st.cache_resource
def load_model():
    path = 'MODELS/best.pt' if os.path.exists('MODELS/best.pt') else 'best.pt'
    if os.path.exists(path):
        return YOLO(path)
    return None

model = load_model()

# صوت تنبيه
def trigger_alert():
    st.components.v1.html("""
        <audio autoplay>
        <source src="https://www.soundjay.com/buttons/beep-01a.mp3">
        </audio>
    """, height=0)

# الواجهة
def main():
    st.title("🛡️ SafeRoad AI")
    st.warning("🚨 SYSTEM ACTIVE")

    if model is None:
        st.error("❌ Model not found (best.pt)")
        return

    img_file = st.camera_input("📷 افتح الكاميرا")

    if img_file:
        bytes_data = img_file.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        results = model.predict(img, conf=0.25, verbose=False)

        if len(results[0].boxes) > 0:
            annotated = results[0].plot()

            trigger_alert()

            st.error("🚨 تم اكتشاف حفرة! سيتم إبلاغ الجهات المختصة")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

            # حفظ الصورة
            img_path = "pothole.jpg"
            cv2.imwrite(img_path, annotated)

            # إنشاء PDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, "SafeRoad Detection Report", ln=True, align='C')
            pdf.image(img_path, x=10, y=40, w=180)

            pdf_path = "report.pdf"
            pdf.output(pdf_path)

            with open(pdf_path, "rb") as f:
                st.download_button("📥 تحميل التقرير", f, "report.pdf")

        else:
            st.success("✅ الطريق سليم")

if __name__ == "__main__":
    main()
