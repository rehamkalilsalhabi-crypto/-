import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
import pandas as pd
from fpdf import FPDF

# 1. Smart Model Loader
@st.cache_resource
def load_yolo_model():
    search_paths = [
        'MODELS/best.pt', 
        'models/best.pt', 
        'best.pt',
        os.path.join(os.getcwd(), 'MODELS', 'best.pt')
    ]
    for path in search_paths:
        if os.path.exists(path):
            return YOLO(path)
    return None

model = load_yolo_model()

# 2. PDF Report Generator
def create_pdf_report(detections_count, image_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="SafeRoad AI - Detection Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Total Potholes Detected: {detections_count}", ln=True, align='L')
    
    if detections_count > 0:
        # Drawing the image in the report
        pdf.image(image_path, x=10, y=50, w=180)
    
    report_name = "Official_Road_Report.pdf"
    pdf.output(report_name)
    return report_name

def render_camera_detection():
    st.title("🎥 Automated Detection & Reporting")
    st.write("---")

    if model is None:
        st.error("❌ Model not found! Ensure 'best.pt' is in the MODELS folder.")
        return

    # Camera Input Component
    img_file = st.camera_input("Scan Road for Defects")

    if img_file is not None:
        # Convert image to OpenCV format
        bytes_data = img_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        with st.spinner('SafeRoad AI is analyzing the road...'):
            # --- التعديل الجوهري هنا ---
            # خفضنا conf لـ 0.2 ليصبح الرصد أكثر حساسية
            results = model.predict(cv2_img, conf=0.2) 
            # ---------------------------
            
            annotated_img = results[0].plot()
            num_detections = len(results[0].boxes)

            # Temporary save for the report
            temp_img_path = "temp_detected_pothole.jpg"
            cv2.imwrite(temp_img_path, annotated_img)

            # Auto-Logging to CSV
            log_entry = {
                "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                "Detections": [num_detections],
                "Status": ["Alert" if num_detections > 0 else "Safe"]
            }
            df = pd.DataFrame(log_entry)
            df.to_csv("detections_log.csv", mode='a', header=not os.path.exists("detections_log.csv"), index=False)

            # Display Results
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Detection Result")
            st.metric("Potholes Found", num_detections)

            # Automatic Reporting Logic
            if num_detections > 0:
                st.warning(f"⚠️ {num_detections} Potholes detected! Creating report...")
                report_file = create_pdf_report(num_detections, temp_img_path)
                
                with open(report_file, "rb") as f:
                    st.download_button(
                        label="📥 Download Official Report (PDF)",
                        data=f,
                        file_name=f"SafeRoad_Report_{datetime.now().strftime('%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
            else:
                st.balloons()
                st.success("✅ No potholes detected. The road is clear!")

if __name__ == "__main__":
    render_camera_detection()
