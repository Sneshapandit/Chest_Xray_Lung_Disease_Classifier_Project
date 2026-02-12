import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directory paths
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
preferred_raw = os.path.join(PROJECT_ROOT, "data", "raw", "pre2kpro")
legacy_raw = os.path.join(PROJECT_ROOT, "pre2kpro-20250116T135723Z-001", "pre2kpro")
input_dir = preferred_raw if os.path.isdir(preferred_raw) else legacy_raw

preferred_processed = os.path.join(PROJECT_ROOT, "data", "processed")
output_dir = preferred_processed
os.makedirs(output_dir, exist_ok=True)

# Diseases of interest
diseases = ["pneumonia", "tuberculosis", "covid19", "normal", "pleural_effusion"]

# Initialize lists to store images and labels
images = []
labels = []

# Preprocess images
for disease in diseases:
    class_dir = os.path.join(input_dir, disease)
    if os.path.isdir(class_dir):
        logging.info(f"Processing {disease}...")
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224), color_mode='rgb')
                img = tf.keras.applications.resnet50.preprocess_input(
                    tf.keras.preprocessing.image.img_to_array(img)
                )
                images.append(img)
                labels.append(disease)
            except Exception as e:
                logging.warning(f"Skipping {img_path}: {e}")

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

if len(images) == 0:
    logging.error("No images were loaded. Exiting.")
    exit()

# Encode labels
label_to_index = {label: idx for idx, label in enumerate(np.unique(labels))}
labels_encoded = np.array([label_to_index[label] for label in labels])

# Normalize images
images = images / 255.0

# Save processed data
np.save(os.path.join(output_dir, "images.npy"), images)
np.save(os.path.join(output_dir, "labels.npy"), labels_encoded)
np.save(os.path.join(output_dir, "label_to_index.npy"), label_to_index)
logging.info(f"Processed data saved to {output_dir}")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

# Save training and testing datasets
train_test_split_path = os.path.join(output_dir, "train_test_split")
os.makedirs(train_test_split_path, exist_ok=True)
np.save(os.path.join(train_test_split_path, "X_train.npy"), X_train)
np.save(os.path.join(train_test_split_path, "y_train.npy"), y_train)
np.save(os.path.join(train_test_split_path, "X_test.npy"), X_test)
np.save(os.path.join(train_test_split_path, "y_test.npy"), y_test)
logging.info(f"Training and testing datasets saved to {train_test_split_path}")

# Step 2: Feature Extraction with ResNet50
resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=resnet.input, outputs=tf.keras.layers.GlobalAveragePooling2D()(resnet.output))

# Extract features
train_features = model.predict(X_train, verbose=1)
test_features = model.predict(X_test, verbose=1)

# Save ResNet50 feature extractor model
resnet_model_path = os.path.join(output_dir, "resnet_feature_extractor.keras")
model.save(resnet_model_path)
logging.info(f"ResNet50 feature extractor model saved to {resnet_model_path}")

# Step 3: Dimensionality Reduction using PCA
pca = PCA(n_components=100)  # Parameterizable
train_features_reduced = pca.fit_transform(train_features)
test_features_reduced = pca.transform(test_features)

# Save PCA model
pca_path = os.path.join(output_dir, "pca_model.pkl")
with open(pca_path, 'wb') as f:
    pickle.dump({'components': pca.components_, 'explained_variance': pca.explained_variance_, 'mean': pca.mean_}, f)
logging.info(f"PCA model saved to {pca_path}")

# Step 4: Build a Classification Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

classifier = Sequential([
    Dense(128, activation='relu', input_shape=(train_features_reduced.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(label_to_index), activation='softmax')
])

classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Handle class imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# Train the classifier
classifier.fit(
    train_features_reduced, y_train,
    epochs=10, batch_size=32,
    validation_data=(test_features_reduced, y_test),
    class_weight=class_weights_dict
)

# Save trained classifier model
classifier_model_path = os.path.join(output_dir, "classifier_model.keras")
classifier.save(classifier_model_path)
logging.info(f"Classifier model saved to {classifier_model_path}")

# Step 5: Evaluate Model and Generate Confusion Matrix
y_pred = np.argmax(classifier.predict(test_features_reduced), axis=1)
conf_matrix = confusion_matrix(y_test, y_pred)


print(f"Total test images: {len(X_test)}")
print(f"Total labels in y_test: {len(y_test)}")
print(f"Total predicted labels in y_pred: {len(y_pred)}")

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_to_index.keys(), yticklabels=label_to_index.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
