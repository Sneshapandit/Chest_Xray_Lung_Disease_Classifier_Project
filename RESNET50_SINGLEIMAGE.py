import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import logging
import pickle

import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from tensorflow.keras.preprocessing import image

logging.basicConfig(level=logging.ERROR)
tf.get_logger().setLevel("ERROR")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
preferred_processed = os.path.join(PROJECT_ROOT, "data", "processed")
legacy_processed = os.path.join(PROJECT_ROOT, "processed_data-20260212T135828Z-1-001", "processed_data")
output_dir = preferred_processed if os.path.isdir(preferred_processed) else legacy_processed

label_to_index = np.load(os.path.join(output_dir, "label_to_index.npy"), allow_pickle=True).item()
index_to_label = {v: k for k, v in label_to_index.items()}

resnet_model_path = os.path.join(output_dir, "resnet_feature_extractor.keras")
feature_extractor = tf.keras.models.load_model(resnet_model_path)

pca_path = os.path.join(output_dir, "pca_model.pkl")
with open(pca_path, "rb") as f:
    pca_data = pickle.load(f)

pca = PCA(n_components=100)
pca.components_ = pca_data["components"]
pca.explained_variance_ = pca_data["explained_variance"]
pca.mean_ = pca_data["mean"]

classifier_model_path = os.path.join(output_dir, "classifier_model.keras")
classifier = tf.keras.models.load_model(classifier_model_path)


def classify_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224), color_mode="rgb")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

    img_features = feature_extractor.predict(img_array, verbose=0)
    img_features_reduced = pca.transform(img_features)

    probs = classifier.predict(img_features_reduced, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    predicted_label = index_to_label[pred_idx]
    confidence = float(np.max(probs))

    print("\n=== Lung Disease Prediction (ResNet50) ===")
    print(f"Image      : {img_path}")
    print(f"Prediction : {predicted_label}")
    print(f"Confidence : {confidence:.2%}")
    print("Status     : SUCCESS\n")
    return predicted_label


def main():
    preferred_raw_img = os.path.join(
        PROJECT_ROOT,
        "data",
        "raw",
        "pre2kpro",
        "covid19",
        "COVID19(26)_preprocessed.jpg",
    )
    legacy_raw_img = os.path.join(
        PROJECT_ROOT,
        "pre2kpro-20250116T135723Z-001",
        "pre2kpro",
        "covid19",
        "COVID19(26)_preprocessed.jpg",
    )
    image_path = preferred_raw_img if os.path.isfile(preferred_raw_img) else legacy_raw_img

    if not os.path.isfile(image_path):
        print("\n=== Lung Disease Prediction (ResNet50) ===")
        print("Status     : FAILED")
        print("Reason     : Sample image not found.")
        print("Action     : Pass a valid path to classify_image(<path>).\n")
        return

    classify_image(image_path)


if __name__ == "__main__":
    main()

