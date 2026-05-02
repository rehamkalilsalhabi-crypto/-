import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os

# 1. حل مشكلة المسار (البحث عن الموديل في المجلد الصحيح)
@st.cache_resource
def load_yolo_model():
    # المسارات المحتملة للموديل بناءً على هيكلة مشروعك
    possible_paths = [
        os.path.join(os.getcwd(), 'models', 'best.pt'), # المسار الأول: models/best.pt
        os.path.join(os.getcwd(), 'best.pt'),          # المسار الثاني: بجانب app.py مباشرة
        'models/best.pt',
        'best.pt'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return YOLO(path)
    
    # إذا لم يجد الملف، سيعيد نصاً للخطأ بدلاً من تعليق الموقع
    return None

model = load_yolo_model()

def render_camera_detection():
    st.title("🎥 رصد الحفر المباشر (SafeRoad AI)")
    st.write("---")
    
    # التأكد من تحميل الموديل بنجاح
    if model is None:
        st.error("❌ عذراً، لم نتمكن من العثور على ملف الموديل 'best.pt'. تأكدي من رفعه إلى GitHub في مجلد models.")
        st.info("💡 نصيحة: تأكدي أن اسم الملف best.pt بحروف صغيرة.")
        return

    st.info("💡 وجهي الكاميرا نحو الطريق والتقطي صورة ليقوم النظام بتحليلها فوراً.")

    # 2. ميزة الكاميرا الأصلية (تفتح كاميرا الجوال مباشرة)
    img_file = st.camera_input("التقطي صورة للطريق")

    if img_file is not None:
        # تحويل الصورة الملتقطة لتنسيق OpenCV
        bytes_data = img_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        with st.spinner('جاري تحليل الصورة ورصد الحفر...'):
            # إجراء الرصد باستخدام YOLOv8
            results = model.predict(cv2_img, conf=0.4)
            
            # رسم المربعات (Bounding Boxes) على الصورة
            annotated_img = results[0].plot()
            
            # تحويل الألوان من BGR إلى RGB للعرض الصحيح في المتصفح
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            st.success("تمت عملية الرصد بنجاح!")
            
            # عرض الصورة الناتجة
            st.image(annotated_img_rgb, caption="نتائج التحليل من SafeRoad AI", use_container_width=True)

            # إظهار عدد الحفر المكتشفة كإحصائية
            num_detections = len(results[0].boxes)
            st.metric(label="عدد الحفر المرصودة في هذه الصورة", value=num_detections)

            if num_detections > 0:
                st.warning(f"⚠️ تم رصد {num_detections} منطقة خلل في الطريق. يرجى الحذر!")
            else:
                st.balloons()
                st.success("✅ لم يتم رصد أي حفر في هذه الصورة. الطريق يبدو جيداً!")

if __name__ == "__main__":
    render_camera_detection()
