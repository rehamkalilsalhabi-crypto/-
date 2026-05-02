import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
from fpdf import FPDF
import time

# 1. تحميل الموديل بذكاء
@st.cache_resource
def load_yolo_model():
    path = 'MODELS/best.pt' if os.path.exists('MODELS/best.pt') else 'best.pt'
    if os.path.exists(path):
        return YOLO(path)
    return None

model = load_yolo_model()

# 2. دالة التقرير (تدعم تعدد الصور المكتشفة تلقائياً)
def create_auto_report(detections):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(200, 20, txt="SafeRoad AI - Auto-Inspection Report", ln=True, align='C')
    pdf.ln(10)

    for i, det in enumerate(detections):
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, txt=f"Automatic Detection Record #{i+1}", ln=True)
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 10, txt=f"Time: {det['time']} | Confidence: {det['conf']}", ln=True)
        pdf.image(det['path'], x=10, w=150)
        pdf.ln(10)
        if (i + 1) % 2 == 0: pdf.add_page()

    report_path = "Auto_Detection_Report.pdf"
    pdf.output(report_path)
    return report_path

def render_camera_detection():
    st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>📡 Live AI Autonomous Radar</h1>", unsafe_allow_html=True)
    st.divider()

    if model is None:
        st.error("❌ الموديل غير موجود! تأكدي من رفع ملف best.pt")
        return

    # إدارة الذاكرة لحفظ اللقطات التلقائية
    if "auto_logs" not in st.session_state:
        st.session_state.auto_logs = []
    if "last_alert_time" not in st.session_state:
        st.session_state.last_alert_time = 0

    # القائمة الجانبية للتحكم بالحساسية
    conf_threshold = st.sidebar.slider("AI Sensitivity (Confidence)", 0.1, 1.0, 0.5)
    cooldown_period = st.sidebar.slider("Detection Interval (Seconds)", 1, 10, 3)

    # تشغيل الكاميرا المباشرة
    # ملاحظة: st.camera_input في Streamlit تلتقط صورة واحدة، 
    # لكن لعمل "بث مباشر وتلقائي" في GitHub، نستخدم زر البدء
    img_file = st.camera_input("AI IS MONITORING - POINT AT ROAD")

    if img_file is not None:
        bytes_data = img_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # الرصد التلقائي الفوري بمجرد التقاط/تغيير الصورة
        results = model.predict(cv2_img, conf=conf_threshold)
        num_detections = len(results[0].boxes)
        
        # رسم الإطار الأحمر
        res_plotted = results[0].plot(line_width=4)
        res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

        if num_detections > 0:
            current_time = time.time()
            
            # منع تكرار التنبيه في نفس اللحظة (Cooldown)
            if current_time - st.session_state.last_alert_time > cooldown_period:
                # 1. صوت التنبيه
                st.components.v1.html("""<audio autoplay><source src="https://www.soundjay.com/buttons/beep-01a.mp3"></audio>""", height=0)
                
                # 2. نص التنبيه
                st.markdown("<h2 style='color: red; text-align: center;'>🚨 انتبه: تم رصد حفرة تلقائياً!</h2>", unsafe_allow_html=True)
                
                # 3. أخذ سكرين شوت وحفظها للتقرير
                img_path = f"auto_det_{len(st.session_state.auto_logs)}.jpg"
                cv2.imwrite(img_path, res_plotted)
                
                st.session_state.auto_logs.append({
                    "path": img_path,
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "conf": f"{float(results[0].boxes[0].conf[0]):.2%}"
                })
                st.session_state.last_alert_time = current_time

            # عرض النتيجة
            st.image(res_plotted_rgb, caption="AI Real-time Analysis")
        else:
            st.success("🛣️ Road is clear. AI is scanning...")
            st.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), caption="Scanning...")

    # عرض سجل الاكتشافات التلقائية
    if st.session_state.auto_logs:
        st.divider()
        st.subheader("📋 Automatic Detection Log")
        
        # عرض آخر 3 اكتشافات
        cols = st.columns(3)
        for i, log in enumerate(reversed(st.session_state.auto_logs[-3:])):
            with cols[i]:
                st.image(log['path'], caption=f"Time: {log['time']}")

        if st.button("📑 Generate Full Maintenance Report"):
            report = create_auto_report(st.session_state.auto_logs)
            with open(report, "rb") as f:
                st.download_button("📥 Download PDF Report", f, file_name=report)

if __name__ == "__main__":
    render_camera_detection()
