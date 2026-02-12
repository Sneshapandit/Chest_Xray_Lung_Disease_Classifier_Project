import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from sklearn.decomposition import PCA
from tensorflow.keras.preprocessing import image
import os

# Project root
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# Paths (prefer new `data/processed`, fall back to legacy locations if needed)
preferred_output = os.path.join(PROJECT_ROOT, "data", "processed")
legacy_output = os.path.join(PROJECT_ROOT, "processed_data-20260212T135828Z-1-001", "processed_data")
output_dir = preferred_output if os.path.isdir(preferred_output) else legacy_output
label_to_index = np.load(os.path.join(output_dir, "label_to_index.npy"), allow_pickle=True).item()
index_to_label = {v: k for k, v in label_to_index.items()}
resnet_model_path = os.path.join(output_dir, "resnet_feature_extractor.keras")
classifier_model_path = os.path.join(output_dir, "classifier_model.keras")
pca_path = os.path.join(output_dir, "pca_model.pkl")

# Load models
feature_extractor = tf.keras.models.load_model(resnet_model_path)
classifier = tf.keras.models.load_model(classifier_model_path)

with open(pca_path, 'rb') as f:
    pca_data = pickle.load(f)
pca = PCA(n_components=100)
pca.components_ = pca_data['components']
pca.explained_variance_ = pca_data['explained_variance']
pca.mean_ = pca_data['mean']

# Classifier function
def classify_image(img):
    img = image.load_img(img, target_size=(224, 224), color_mode='rgb')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    features = feature_extractor.predict(img_array)
    reduced_features = pca.transform(features)
    prediction = np.argmax(classifier.predict(reduced_features), axis=1)[0]
    return index_to_label[prediction]

# Streamlit UI
st.set_page_config(page_title="Lung Disease Classifier", layout="centered")
st.title("Lung Disease detection using AI")

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    if st.button("Classify"):
        with st.spinner("Analyzing..."):
            result = classify_image(uploaded_file)
        st.success(f"Predicted Disease: **{result}**")
