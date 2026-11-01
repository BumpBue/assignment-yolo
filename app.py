import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np

st.set_page_config(page_title="Traffic & Vehicle Detection", page_icon="🚦", layout="wide")

@st.cache_resource
def load_models():
    bbox_model = YOLO('best_bbox.pt')
    poly_model = YOLO('best_polygon.pt')
    return bbox_model, poly_model

model_bbox, model_poly = load_models()

CLASS_NAMES = model_bbox.names

st.sidebar.title("ตั้งค่าระบบประมวลผล")
uploaded_file = st.sidebar.file_uploader("อัปโหลดรูปภาพ (JPG, PNG)", type=['jpg', 'jpeg', 'png'])

conf_threshold = st.sidebar.slider("ความแม่นยำขั้นต่ำ (Confidence)", min_value=0.0, max_value=1.0, value=0.25, step=0.05)

display_mode = st.sidebar.radio(
    "เลือกรูปแบบการตรวจจับ:",
    ("ตีกรอบสี่เหลี่ยม (Object Detection)", "ระบายสีทับ (Instance Segmentation)")
)

st.sidebar.markdown("---")
st.sidebar.subheader("เลือกวัตถุที่ต้องการแสดงผล")
default_classes = ['car', 'truck', 'bus', 'trafficlight', 'road']
selected_class_names = st.sidebar.multiselect(
    "กรองประเภทวัตถุ:", 
    options=list(CLASS_NAMES.values()), 
    default=default_classes
)

selected_class_indices = [k for k, v in CLASS_NAMES.items() if v in selected_class_names]

st.title("ระบบตรวจจับยานพาหนะและสภาพถนนอัจฉริยะ")
st.markdown("ระบบวิเคราะห์ภาพด้วย AI (YOLOv8) รองรับการตีกรอบ Bounding Box และระบายสี Polygon")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 ภาพต้นฉบับ")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("ผลลัพธ์จาก AI")
        
        model = model_bbox if display_mode == "ตีกรอบสี่เหลี่ยม (Object Detection)" else model_poly

        if len(selected_class_indices) > 0:
            results = model.predict(source=image, conf=conf_threshold, classes=selected_class_indices)
            res = results[0]

            plotted_img = res.plot()
            st.image(plotted_img, use_column_width=True)
            
            st.markdown("---")
            st.subheader("สรุปจำนวนวัตถุที่ตรวจพบ")

            detected_classes = res.boxes.cls.cpu().numpy()
            
            metric_cols = st.columns(len(selected_class_names))
            
            for idx, class_name in enumerate(selected_class_names):
                class_id = list(CLASS_NAMES.keys())[list(CLASS_NAMES.values()).index(class_name)]
                count = np.count_nonzero(detected_classes == class_id)
                
                with metric_cols[idx]:
                    st.metric(label=class_name.capitalize(), value=count)
        else:
            st.warning("กรุณาเลือกประเภทวัตถุที่ต้องการตรวจจับอย่างน้อย 1 รายการจากแถบด้านซ้าย")
else:
    st.info("กรุณาอัปโหลดรูปภาพที่แถบเมนูด้านซ้ายเพื่อเริ่มต้นใช้งาน")