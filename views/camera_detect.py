import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
import pandas as pd
from fpdf import FPDF

# 1. Load YOLOv8 Model with Path Safety
@st.cache_resource
def load_yolo_model():
    search_paths = ['MODELS/best.pt', 'models/best.pt', 'best.pt']
    for path in search_paths:
        if os.path.exists(path):
            return YOLO(path)
    return None

model = load_yolo_model()

# 2. PDF Report Logic
def create_pdf_report(detections_count, image_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="SafeRoad AI - Automated Detection Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Status: Potholes Detected", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Count: {detections_count}", ln=True, align='L')
    pdf.image(image_path, x=10, y=50, w=180)
    
    report_name = "Automated_Road_Report.pdf"
    pdf.output(report_name)
    return report_name

# 3. Real-time Video Processor
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_capture = None
        self.pothole_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Automatic Detection with sensitive threshold (0.2)
        results = model.predict(img, conf=0.2, verbose=False)
        annotated_img = results[0].plot()
        
        count = len(results[0].boxes)
        if count > 0:
            self.last_capture = annotated_img
            self.pothole_count = count
            
        return frame.from_ndarray(annotated_img, format="bgr24")

def render_camera_detection():
    st.title("🤖 Autonomous Pothole Detection")
    st.info("The system is now monitoring. It will automatically detect and prepare reports.")

    if model is None:
        st.error("Model 'best.pt' not found in MODELS folder.")
        return

    # WebRTC Streamer for Continuous Monitoring
    ctx = webrtc_streamer(
        key="auto-detect",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
    )

    # 4. Auto-Report Generation UI
    if ctx.video_processor:
        if ctx.video_processor.last_capture is not None:
            st.write("---")
            st.warning(f"🚨 System detected {ctx.video_processor.pothole_count} potholes!")
            
            # Save the captured frame for the report
            temp_path = "auto_captured_pothole.jpg"
            cv2.imwrite(temp_path, ctx.video_processor.last_capture)
            
            # Generate Report Button
            report_file = create_pdf_report(ctx.video_processor.pothole_count, temp_path)
            
            with open(report_file, "rb") as f:
                st.download_button(
                    label="📥 Download Detection Report",
                    data=f,
                    file_name=f"SafeRoad_Auto_Report.pdf",
                    mime="application/pdf"
                )
            
            # Optional: Log to CSV
            if st.button("Save to Logs"):
                log_df = pd.DataFrame({"Time": [datetime.now()], "Count": [ctx.video_processor.pothole_count]})
                log_df.to_csv("detections_log.csv", mode='a', index=False)
                st.success("Log Updated!")

if __name__ == "__main__":
    render_camera_detection()
