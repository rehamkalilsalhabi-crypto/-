import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# تحميل الموديل
@st.cache_resource
def load_yolo_model():
    return YOLO('models/best.pt')

model = load_yolo_model()

def render_camera_detection():
    st.title("📸 التقاط ورصد الحفر (SafeRoad AI)")
    st.write("---")
    
    st.info("💡 التقط صورة للطريق الآن ليقوم الذكاء الاصطناعي بفحصها.")

    # فتح كاميرا الجوال مباشرة (ميزة مدمجة في ستريمليت)
    img_file = st.camera_input("وجه الكاميرا نحو الطريق واضغط 'Take Photo'")

    if img_file is not None:
        # تحويل الصورة الملتقطة لتنسيق يفهمه YOLO
        bytes_data = img_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        with st.spinner('جاري تحليل الصورة ورصد الحفر...'):
            # إجراء الرصد
            results = model.predict(cv2_img, conf=0.4)
            
            # رسم النتائج
            annotated_img = results[0].plot()
            
            # تحويلها من BGR إلى RGB للعرض في ستريمليت
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            st.success("تمت عملية الرصد بنجاح!")
            st.image(annotated_img_rgb, caption="نتائج رصد الحفر", use_container_width=True)

            # إحصائيات بسيطة للفريق
            detections = len(results[0].boxes)
            st.metric(label="عدد الحفر المرصودة", value=detections)
