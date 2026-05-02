import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os

# 1. دالة ذكية لتحميل الموديل والبحث عنه في كافة المجلدات
@st.cache_resource
def load_yolo_model():
    # قائمة بالمسارات المحتملة للموديل (بناءً على أخطاء المسارات الشائعة)
    search_paths = [
        'models/best.pt',
        'best.pt',
        os.path.join(os.getcwd(), 'models', 'best.pt'),
        os.path.join(os.getcwd(), 'best.pt')
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return YOLO(path)
    
    return None

model = load_yolo_model()

def render_camera_detection():
    st.title("🎥 رصد الحفر المباشر (SafeRoad AI)")
    st.write("---")
    
    # 2. فحص وجود الموديل وإظهار تعليمات للمستخدم في حال عدم وجوده
    if model is None:
        st.error("❌ لم يتم العثور على ملف الموديل 'best.pt'")
        st.write("### 🛠️ ماذا تفعلين الآن؟")
        st.write("1. تأكدي من وجود مجلد باسم **models** في GitHub.")
        st.write("2. تأكدي أن الملف بداخله واسمه **best.pt** (حروف صغيرة).")
        
        # كود لمساعدتك في رؤية الملفات المرفوعة فعلياً
        with st.expander("🔍 اضغطي هنا لرؤية الملفات المرفوعة حالياً على السيرفر"):
            st.write("الملفات في المجلد الرئيسي:")
            st.code(os.listdir('.'))
            if os.path.exists('models'):
                st.write("الملفات داخل مجلد models:")
                st.code(os.listdir('models'))
        return

    st.success("✅ الموديل جاهز للعمل!")
    st.info("💡 وجهي الكاميرا نحو الطريق والتقطي صورة ليتم فحصها فوراً.")

    # 3. استخدام أداة الكاميرا الأصلية من Streamlit
    img_file = st.camera_input("التقطي صورة للطريق")

    if img_file is not None:
        # تحويل الصورة الملتقطة إلى تنسيق OpenCV
        bytes_data = img_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        with st.spinner('جاري تحليل الصورة ورصد الحفر باستخدام YOLOv8...'):
            # إجراء الرصد
            results = model.predict(cv2_img, conf=0.4)
            
            # رسم النتائج على الصورة
            annotated_img = results[0].plot()
            
            # تحويل الألوان من BGR إلى RGB للعرض في المتصفح
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            st.success("تمت عملية الرصد بنجاح!")
            st.image(annotated_img_rgb, caption="نتائج رصد SafeRoad AI", use_container_width=True)

            # إظهار الإحصائيات (عدد الحفر)
            num_detections = len(results[0].boxes)
            st.metric(label="عدد الحفر المرصودة", value=num_detections)

            if num_detections > 0:
                st.warning(f"⚠️ انتبه: تم رصد {num_detections} منطقة خلل في الطريق.")
            else:
                st.balloons()
                st.success("✅ الطريق يبدو آمناً، لم يتم رصد أي حفر!")

if __name__ == "__main__":
    render_camera_detection()
