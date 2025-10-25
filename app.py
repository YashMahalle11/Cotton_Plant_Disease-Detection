import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# -------------------------------
# Must be the first Streamlit command
# -------------------------------
st.set_page_config(page_title="Cotton Disease Detection", page_icon="🌱", layout="centered")

# -------------------------------
# App title and description
# -------------------------------
st.title("🌱 Cotton Disease Detection (ResNet-50)")
st.markdown(
    """
    Upload a cotton leaf image, and the model will predict the most likely disease.  
    The model is based on **ResNet-50** trained on a cotton leaf dataset.
    """
)

# -------------------------------
# Load and cache the trained model
# -------------------------------
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(__file__)
    MODEL_PATH = os.path.join(BASE_DIR, "resnet50.h5")  # Ensure file is in project root
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

st.write("Model output shape:", model.output_shape)

# -------------------------------
# Define the disease class labels
# -------------------------------
class_names = [
    "Aphids",
    "Army Worm",
    "Bacterial Blight",
    "Powdery Mildew",
    "Target Spot",
    "Healthy"
]

# -------------------------------
# File uploader
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload a cotton leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))  # ResNet-50 expects 224x224
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    preds = model.predict(img_array)[0]
    top_idx = np.argmax(preds)

    if top_idx < len(class_names):
        pred_class = class_names[top_idx]
        confidence = float(tf.nn.softmax(preds)[top_idx] * 100)

        st.success(f"✅ Prediction: **{pred_class}** ({confidence:.2f}% confidence)")

        # Show probabilities for all classes
        st.subheader("📊 Prediction Probabilities")
        for i, cls in enumerate(class_names):
            st.write(f"- {cls}: {100 * preds[i]:.2f}%")
    else:
        st.error("⚠ Model output size and class_names length mismatch. Please update class_names.")
