import streamlit as st
from PIL import Image
from utils import load_model, preprocess_image, predict

st.set_page_config(page_title="Pneumonia Detection")

st.title("pneumonia Detection from Chest X-Ray")

@st.cache_resource
def get_model():
    return load_model()

model = get_model()

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    image_tensor = preprocess_image(image)
    label, confidence = predict(model, image_tensor)

    st.subheader("Prediction:")
    st.success(label)
    st.write(f"Confidence: {confidence*100:.2f}%")