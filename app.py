import streamlit as st
from PIL import Image
from utils import load_model, preprocess_image, predict

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Pneumonia Detection AI",
    layout="centered"
)

# ----------------------------
# Custom Styling
# ----------------------------
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: #1f4e79;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: gray;
            margin-bottom: 30px;
        }
        .author {
            text-align: center;
            font-size: 14px;
            color: #888;
            margin-top: 50px;
        }
        .result-box {
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------
st.markdown('<div class="main-title">Pneumonia Detection from Chest X-Ray</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Deep Learning powered Medical AI System</div>', unsafe_allow_html=True)

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# ----------------------------
# Upload Section
# ----------------------------
st.markdown("### 📤 Upload Chest X-Ray Image")
uploaded_file = st.file_uploader("Uploade Chest x-ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded X-Ray", width="stretch")

    image_tensor = preprocess_image(image)
    label, confidence = predict(model, image_tensor)

    st.markdown("### 🔍 Prediction Result")

    if label == "Pneumonia":
        st.error(f"⚠️ {label}")
    else:
        st.success(f"✅ {label}")

    st.info(f"Confidence: {confidence*100:.2f}%")

# ----------------------------
# Footer
# ----------------------------
st.markdown('<div class="author">Author: Thanuja</div>', unsafe_allow_html=True)

