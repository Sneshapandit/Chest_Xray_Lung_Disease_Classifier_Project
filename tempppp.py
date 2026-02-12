import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Set constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 10
BASE_DIR = "dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

# Step 1: Preprocessing function
def preprocess_image(image_path):
    """Preprocess images to constant shape."""
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    return img_array

# Step 2: Load and preprocess dataset
def load_dataset(base_dir):
    """Load and preprocess dataset, then split into train and test sets."""
    images, labels = [], []
    label_map = {"covid": 0, "pneumonia": 1, "tb": 2, "pleural": 3, "normal": 4}
    for label, index in label_map.items():
        folder_path = os.path.join(base_dir, label)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if file.endswith(".jpg") or file.endswith(".png"):
                images.append(preprocess_image(file_path))
                labels.append(index)
    return np.array(images), np.array(labels)

# Load train and test datasets
X, y = load_dataset(BASE_DIR)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Save processed images
def save_processed_images(X, y, output_dir):
    """Save processed images to specified directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, (img, label) in enumerate(zip(X, y)):
        label_dir = os.path.join(output_dir, str(label))
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        file_path = os.path.join(label_dir, f"img_{i}.jpg")
        tf.keras.preprocessing.image.save_img(file_path, img)

save_processed_images(X_train, y_train, TRAIN_DIR)
save_processed_images(X_test, y_test, TEST_DIR)

# Step 4: Model definitions
# CNN Model for classification
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# VGG16 Model for feature extraction and classification
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
for layer in vgg_base.layers:
    layer.trainable = False

vgg_model = Sequential([
    vgg_base,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])
vgg_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 5: Training the models
print("Training CNN Model...")
cnn_model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

print("Training VGG16 Model...")
vgg_model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

# Step 6: Evaluate the models
print("Evaluating CNN Model...")
cnn_predictions = np.argmax(cnn_model.predict(X_test), axis=1)
print(classification_report(y_test, cnn_predictions))

print("Evaluating VGG16 Model...")
vgg_predictions = np.argmax(vgg_model.predict(X_test), axis=1)
print(classification_report(y_test, vgg_predictions))

# Save the models
cnn_model.save("cnn_model_for_classification.h5")
vgg_model.save("vgg16_model_for_classification.h5")