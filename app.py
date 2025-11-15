import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# ===============================
# LOAD MODEL
# ===============================
model = tf.keras.models.load_model("asl_model.h5")

# 29 labels (A–Z + 3 special signs)
labels = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "SPACE", "DELETE", "NOTHING"
]

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="ASL Detection", page_icon="🖐")
st.title("🖐 American Sign Language Detection")
st.markdown("Upload an ASL image to identify the hand sign.")

file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((64, 64))
    img_array = np.expand_dims(img, axis=0) / 255.0

    pred = model.predict(img_array)
    label = labels[np.argmax(pred)]

    st.success(f"### ✅ Predicted Sign: {label}")
