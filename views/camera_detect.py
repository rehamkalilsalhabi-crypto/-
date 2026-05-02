import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os

# 1. Smart model loader
@st.cache_resource
def load_yolo_model():
    search_paths = [
        'MODELS/best.pt',
        'models/best.pt',
        'best.pt',
        os.path.join(os.getcwd(), 'MODELS', 'best.pt'),
        os.path.join(os.getcwd(), 'models', 'best.pt')
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return YOLO(path)
    return None

model = load_yolo_model()

def render_camera_detection():
    # Page Title
    st.title("🎥 Live Pothole Detection (SafeRoad AI)")
    st.write("---")
    
    # Check if model is loaded
    if model is None:
        st.error("❌ Model file 'best.pt' not found!")
        st.info("Make sure the file is uploaded to the 'MODELS' folder in GitHub.")
        return

    st.success("✅ Model loaded successfully!")
    st.info("💡 Point your camera at the road and take a photo for analysis.")

    # 2. Streamlit Camera Input
    img_file = st.camera_input("Take a photo of the road")

    if img_file is not None:
        # Convert image to OpenCV format
        bytes_data = img_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        with st.spinner('Analyzing image with YOLOv8...'):
            # Run Inference
            results = model.predict(cv2_img, conf=0.4)
            
            # Plot Results
            annotated_img = results[0].plot()
            
            # Convert BGR to RGB for Streamlit display
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            st.success("Analysis Complete!")
            st.image(annotated_img_rgb, caption="SafeRoad AI Detection Results", use_container_width=True)

            # Display Stats
            num_detections = len(results[0].boxes)
            st.metric(label="Detected Potholes", value=num_detections)

            if num_detections > 0:
                st.warning(f"⚠️ Warning: {num_detections} road defects detected. Drive safely!")
            else:
                st.balloons()
                st.success("✅ Road looks clear! No potholes detected.")

if __name__ == "__main__":
    render_camera_detection()
