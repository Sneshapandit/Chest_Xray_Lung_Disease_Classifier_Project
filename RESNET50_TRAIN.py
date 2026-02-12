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
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

output_dir = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(output_dir, exist_ok=True)

plots_dir = os.path.join(PROJECT_ROOT, "outputs", "plots")
os.makedirs(plots_dir, exist_ok=True)

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

X_train_full, X_test, y_train_full, y_test = train_test_split(
    images,
    labels_encoded,
    test_size=0.2,
    random_state=SEED,
    stratify=labels_encoded,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.1,
    random_state=SEED,
    stratify=y_train_full,
)

train_test_split_path = os.path.join(output_dir, "train_test_split")
os.makedirs(train_test_split_path, exist_ok=True)
np.save(os.path.join(train_test_split_path, "X_train.npy"), X_train_full)
np.save(os.path.join(train_test_split_path, "y_train.npy"), y_train_full)
np.save(os.path.join(train_test_split_path, "X_test.npy"), X_test)
np.save(os.path.join(train_test_split_path, "y_test.npy"), y_test)
logging.info("Training/testing arrays saved to %s", train_test_split_path)

train_gen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
    rotation_range=12,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.12,
    horizontal_flip=True,
)
val_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)

train_flow = train_gen.flow(X_train, y_train, batch_size=32, shuffle=True, seed=SEED)
val_flow = val_gen.flow(X_val, y_val, batch_size=32, shuffle=False)

base_model = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D(name="gap")(x)
x = Dropout(0.3, name="dropout_head")(x)
out = Dense(len(label_to_index), activation="softmax", name="class_head")(x)

fine_tune_model = Model(inputs=base_model.input, outputs=out, name="resnet50_finetune")
fine_tune_model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

callbacks_stage1 = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6),
]

logging.info("Stage 1 fine-tuning: training classification head with frozen ResNet backbone")
fine_tune_model.fit(
    train_flow,
    epochs=8,
    validation_data=val_flow,
    class_weight=class_weights_dict,
    callbacks=callbacks_stage1,
    verbose=1,
)

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

fine_tune_model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks_stage2 = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
]

logging.info("Stage 2 fine-tuning: unfreezing top ResNet blocks")
fine_tune_model.fit(
    train_flow,
    epochs=18,
    validation_data=val_flow,
    class_weight=class_weights_dict,
    callbacks=callbacks_stage2,
    verbose=1,
)

feature_model = Model(inputs=fine_tune_model.input, outputs=fine_tune_model.get_layer("gap").output)
resnet_model_path = os.path.join(output_dir, "resnet_feature_extractor.keras")
feature_model.save(resnet_model_path)
logging.info("Feature extractor saved to %s", resnet_model_path)

X_train_pp = tf.keras.applications.resnet50.preprocess_input(X_train_full.copy())
X_test_pp = tf.keras.applications.resnet50.preprocess_input(X_test.copy())

train_features = feature_model.predict(X_train_pp, verbose=1)
test_features = feature_model.predict(X_test_pp, verbose=1)

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

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
]

history = classifier.fit(
    train_features_reduced,
    y_train_full,
    epochs=40,
    batch_size=32,
    validation_data=(test_features_reduced, y_test),
    class_weight=dict(enumerate(compute_class_weight(class_weight="balanced", classes=np.unique(y_train_full), y=y_train_full))),
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

print("\n=== ResNet50 Classifier Metrics ===")
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
plt.title("ResNet50 + PCA Classifier Confusion Matrix")
plt.tight_layout()
cm_out = os.path.join(plots_dir, "resnet_confusion_matrix.png")
plt.savefig(cm_out)
plt.close()
logging.info("Confusion matrix saved to %s", cm_out)
