import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import shutil

# Directory paths
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
preferred_raw = os.path.join(PROJECT_ROOT, "data", "raw", "pre2kpro")
legacy_raw = os.path.join(PROJECT_ROOT, "pre2kpro-20250116T135723Z-001", "pre2kpro")
input_dir = preferred_raw if os.path.isdir(preferred_raw) else legacy_raw

preferred_processed = os.path.join(PROJECT_ROOT, "data", "processed")
output_dir = preferred_processed
os.makedirs(output_dir, exist_ok=True)

# Diseases of interest
diseases = ["pneumonia", "tuberculosis", "covid19", "normal"]

# Load and preprocess images
images = []
labels = []

# Create directories for each disease in the output directory
processed_images_dir = os.path.join(output_dir, "images")
os.makedirs(processed_images_dir, exist_ok=True)

for disease in diseases:
    # Create a subdirectory for each disease label
    disease_dir = os.path.join(processed_images_dir, disease)
    os.makedirs(disease_dir, exist_ok=True)
    
    class_dir = os.path.join(input_dir, disease)
    if os.path.isdir(class_dir):
        print(f"Processing {disease}...")
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224), color_mode='rgb')
                img = tf.keras.preprocessing.image.img_to_array(img)
                images.append(img)
                labels.append(disease)
                
                # Save the image into the respective disease folder
                save_path = os.path.join(disease_dir, img_name)
                tf.keras.preprocessing.image.save_img(save_path, img)
            except Exception as e:
                print(f"Skipping {img_path}: {e}")
    else:
        print(f"Directory not found for {disease}, skipping.")

# Convert to numpy arrays
images = np.array(images)
labels = np.array(labels)

if len(images) == 0:
    print("No images were loaded. Exiting.")
    exit()

# Encode labels
label_to_index = {label: idx for idx, label in enumerate(np.unique(labels))}
labels_encoded = np.array([label_to_index[label] for label in labels])

# Normalize images
images = images / 255.0

# Save processed data
np.save(os.path.join(output_dir, "images.npy"), images)
np.save(os.path.join(output_dir, "labels.npy"), labels_encoded)

print(f"Processed data saved to {output_dir}")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

# Step 2: Feature Extraction with ResNet50
resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=resnet.input, outputs=tf.keras.layers.GlobalAveragePooling2D()(resnet.output))

# Extract features
train_features = model.predict(X_train, batch_size=32)
test_features = model.predict(X_test, batch_size=32)

# Step 3: Dimensionality Reduction using PCA
pca = PCA(n_components=100)  # Reduce to 100 features
train_features_reduced = pca.fit_transform(train_features)
test_features_reduced = pca.transform(test_features)

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

# Train the classifier
classifier.fit(train_features_reduced, y_train, epochs=10, batch_size=32, validation_data=(test_features_reduced, y_test))

# Step 5: Evaluate the Model
y_pred = np.argmax(classifier.predict(test_features_reduced), axis=1)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(tf.keras.utils.to_categorical(y_test, len(label_to_index)), 
                        tf.keras.utils.to_categorical(y_pred, len(label_to_index)), 
                        multi_class='ovr')

print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"AUC-ROC: {roc_auc:.2f}")

# Save PCA components and attributes
pca_data = {
    'components': pca.components_,  # The principal components (eigenvectors)
    'explained_variance': pca.explained_variance_,  # Explained variance for each component
    'mean': pca.mean_,  # Mean of each feature (used for transforming new data)
}

# Define the path to save PCA data
pca_path = os.path.join(output_dir, "pca_model.npy")

# Save the PCA data as a dictionary in .npy format
np.save(pca_path, pca_data)
print(f"PCA model saved to {pca_path}")

# Save the model
model.save('trainmodel.keras')
print("Model saved as trainmodel.keras")

# Save label-to-index mapping
np.save(os.path.join(output_dir, "label_to_index.npy"), label_to_index)
print(f"Label-to-index mapping saved to {os.path.join(output_dir, 'label_to_index.npy')}")

model.save('trainmodel.keras')
print("Model saved as model.keras")