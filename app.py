# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 12:52:13 2025

@author: adhir
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("cnn_model.keras")

# CIFAR-10 class labels
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Streamlit App UI
st.set_page_config(page_title="Image Classifier", layout="centered")
st.title("🧠 Image Classifier using Deep Learning")
st.write("Upload an image, and the AI will predict what it is!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image for prediction
    image = image.resize((32, 32))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Show result
    st.success(f"Prediction: **{predicted_class}** ({confidence:.2f}% confidence)")
