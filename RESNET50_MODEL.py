import os
import numpy as np
import tensorflow as tf
import pickle
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directory paths
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
preferred_processed = os.path.join(PROJECT_ROOT, "data", "processed")
legacy_processed = os.path.join(PROJECT_ROOT, "processed_data-20260212T135828Z-1-001", "processed_data")
output_dir = preferred_processed if os.path.isdir(preferred_processed) else legacy_processed
train_test_split_path = os.path.join(output_dir, "train_test_split")

# Load test dataset
X_test = np.load(os.path.join(train_test_split_path, "X_test.npy"))
y_test = np.load(os.path.join(train_test_split_path, "y_test.npy"))

# Load ResNet50 feature extractor model
resnet_model_path = os.path.join(output_dir, "resnet_feature_extractor.keras")
feature_extractor = tf.keras.models.load_model(resnet_model_path)
logging.info("Loaded ResNet50 feature extractor model.")

# Extract features from test images
test_features = feature_extractor.predict(X_test, verbose=1)

# Load PCA model
pca_path = os.path.join(output_dir, "pca_model.pkl")
with open(pca_path, 'rb') as f:
    pca_data = pickle.load(f)
pca = PCA(n_components=100)
pca.components_ = pca_data['components']
pca.explained_variance_ = pca_data['explained_variance']
pca.mean_ = pca_data['mean']

# Transform test features using PCA
test_features_reduced = pca.transform(test_features)
logging.info("Applied PCA transformation on test features.")

# Load trained classifier model
classifier_model_path = os.path.join(output_dir, "classifier_model.keras")
classifier = tf.keras.models.load_model(classifier_model_path)
logging.info("Loaded trained classifier model.")

# Predict test labels
y_pred = np.argmax(classifier.predict(test_features_reduced), axis=1)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Testing Accuracy: {accuracy:.4f}")

# Generate classification report
class_report = classification_report(y_test, y_pred)
logging.info("Classification Report:\n" + class_report)

print(f"Testing Accuracy: {accuracy:.4f}")
print("Classification Report:\n", class_report)
