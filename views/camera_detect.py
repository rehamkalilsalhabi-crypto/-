import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os

# 1. دالة ذكية تبحث عن الموديل في المجلدات (سواء كانت MODELS أو models)
@st.cache_resource
def load_yolo_model():
    # قائمة بالمسارات بناءً على ما ظهر في صورتك (MODELS بحروف كبيرة)
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
    st.title("🎥 رصد الحفر المباشر (SafeRoad AI)")
    st.write("---")
    
    # 2. فحص وجود الموديل وإظهار قائمة الملفات للمساعدة في حال الخطأ
    if model is None:
        st.error("❌ لم يتم العثور على ملف الموديل 'best.pt'")
        st.info("💡 تأكدي أن الملف موجود داخل مجلد MODELS وأن اسمه best.pt بحروف صغيرة.")
        
        with st.expander("🔍 تفاصيل الملفات على السيرفر (للمهندسة ريهام)"):
            st.write("المجلد الرئيسي يحتوي على:")
            st.code(os.listdir('.'))
            # البحث عن مجلد الموديلات أياً كان اسمه
            for folder in os.listdir('.'):
                if folder.lower() == 'models':
                    st.write(f"المحتويات داخل مجلد {folder}:")
                    st.code(os.listdir(folder))
        return

    st.success("✅ تم تحميل الموديل بنجاح!")
    st.info("💡 وجهي الكاميرا نحو الطريق والتقطي صورة للفحص.")

    # 3. استخدام أداة الكاميرا المدمجة
    img_file = st.camera_input("التقطي صورة للطريق")

    if img_file is not None:
        # تحويل الصورة إلى تنسيق OpenCV
        bytes_data = img_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        with st.spinner('جاري تحليل الصورة...'):
            # إجراء الرصد
            results = model.predict(cv2_img, conf=0.4)
            
            # رسم النتائج
            annotated_img = results[0].plot()
            
            # تحويل الألوان للعرض في ستريمليت
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            st.success("تمت عملية الرصد!")
            st.image(annotated_img_rgb, caption="نتائج رصد SafeRoad AI", use_container_width=True)

            # عرض الإحصائيات
            num_detections = len(results[0].boxes)
            st.metric(label="عدد الحفر المرصودة", value=num_detections)

            if num_detections > 0:
                st.warning(f"⚠️ تنبيه: تم رصد {num_detections} منطقة تضرر.")
            else:
                st.balloons()
                st.success("✅ الطريق يبدو سليماً!")

if __name__ == "__main__":
    render_camera_detection()
