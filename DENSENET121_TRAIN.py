import os
import random
import pickle
import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Sequential
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
preferred_raw = os.path.join(PROJECT_ROOT, "data", "raw", "pre2kpro")
legacy_raw = os.path.join(PROJECT_ROOT, "pre2kpro-20250116T135723Z-001", "pre2kpro")
input_dir = preferred_raw if os.path.isdir(preferred_raw) else legacy_raw

output_dir = os.path.join(PROJECT_ROOT, "data", "processed", "densenet")
os.makedirs(output_dir, exist_ok=True)

plots_dir = os.path.join(PROJECT_ROOT, "outputs", "plots")
os.makedirs(plots_dir, exist_ok=True)

# Keep canonical class order to avoid label-index drift across runs.
diseases = ["pneumonia", "tuberculosis", "covid19", "normal", "pleural_effusion"]

images = []
labels = []

for disease in diseases:
    class_dir = os.path.join(input_dir, disease)
    if not os.path.isdir(class_dir):
        logging.warning("Missing class directory: %s", class_dir)
        continue

    logging.info("Processing %s...", disease)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        try:
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224), color_mode="rgb")
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = tf.keras.applications.densenet.preprocess_input(img)
            images.append(img)
            labels.append(disease)
        except Exception as exc:
            logging.warning("Skipping %s: %s", img_path, exc)

images = np.array(images, dtype=np.float32)
labels = np.array(labels)

if len(images) == 0:
    raise RuntimeError("No images were loaded. Aborting training.")

present_labels = sorted(np.unique(labels))
label_to_index = {label: idx for idx, label in enumerate(present_labels)}
labels_encoded = np.array([label_to_index[label] for label in labels])

np.save(os.path.join(output_dir, "images.npy"), images)
np.save(os.path.join(output_dir, "labels.npy"), labels_encoded)
np.save(os.path.join(output_dir, "label_to_index.npy"), label_to_index)
logging.info("Processed data saved to %s", output_dir)

X_train, X_test, y_train, y_test = train_test_split(
    images,
    labels_encoded,
    test_size=0.2,
    random_state=SEED,
    stratify=labels_encoded,
)

train_test_split_path = os.path.join(output_dir, "train_test_split")
os.makedirs(train_test_split_path, exist_ok=True)
np.save(os.path.join(train_test_split_path, "X_train.npy"), X_train)
np.save(os.path.join(train_test_split_path, "y_train.npy"), y_train)
np.save(os.path.join(train_test_split_path, "X_test.npy"), X_test)
np.save(os.path.join(train_test_split_path, "y_test.npy"), y_test)
logging.info("Training/testing arrays saved to %s", train_test_split_path)

base_model = DenseNet121(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
base_model.trainable = False
feature_model = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))

train_features = feature_model.predict(X_train, verbose=1)
test_features = feature_model.predict(X_test, verbose=1)

densenet_model_path = os.path.join(output_dir, "densenet_feature_extractor.keras")
feature_model.save(densenet_model_path)
logging.info("Feature extractor saved to %s", densenet_model_path)

pca = PCA(n_components=100, random_state=SEED)
train_features_reduced = pca.fit_transform(train_features)
test_features_reduced = pca.transform(test_features)

pca_path = os.path.join(output_dir, "pca_model.pkl")
with open(pca_path, "wb") as f:
    pickle.dump(
        {
            "components": pca.components_,
            "explained_variance": pca.explained_variance_,
            "mean": pca.mean_,
        },
        f,
    )
logging.info("PCA model saved to %s", pca_path)

classifier = Sequential(
    [
        Input(shape=(train_features_reduced.shape[1],)),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(len(label_to_index), activation="softmax"),
    ]
)

classifier.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
]

history = classifier.fit(
    train_features_reduced,
    y_train,
    epochs=40,
    batch_size=32,
    validation_data=(test_features_reduced, y_test),
    class_weight=class_weights_dict,
    callbacks=callbacks,
    verbose=1,
)

classifier_model_path = os.path.join(output_dir, "classifier_model.keras")
classifier.save(classifier_model_path)
logging.info("Classifier saved to %s", classifier_model_path)

y_probs = classifier.predict(test_features_reduced, verbose=0)
y_pred = np.argmax(y_probs, axis=1)

acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)

print("\n=== DenseNet121 Classifier Metrics ===")
print(f"Accuracy       : {acc:.4f}")
print(f"Macro F1       : {macro_f1:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall   : {macro_recall:.4f}")
print(f"Best Val Acc   : {max(history.history.get('val_accuracy', [0.0])):.4f}")

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
ordered_labels = [lbl for lbl, _ in sorted(label_to_index.items(), key=lambda x: x[1])]
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=ordered_labels, yticklabels=ordered_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("DenseNet121 + PCA Classifier Confusion Matrix")
plt.tight_layout()
cm_out = os.path.join(plots_dir, "densenet_confusion_matrix.png")
plt.savefig(cm_out)
plt.close()
logging.info("Confusion matrix saved to %s", cm_out)
