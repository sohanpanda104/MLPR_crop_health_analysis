import os
import numpy as np
import pandas as pd
import rasterio
from tensorflow.keras.utils import Sequence, to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.image import resize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
import matplotlib.pyplot as plt

# ========== CONFIG ========== #
CSV_PATH = "/home/bhavya/Downloads/output.csv"
IMAGE_FOLDER = "/home/bhavya/Downloads/downloads"
IMG_SIZE = (128, 128)
NUM_CLASSES = 4
BATCH_SIZE = 8
CHANNELS = 16

# ========== Load Dataset ========== #
original_df = pd.read_csv(CSV_PATH)
original_df = original_df.replace(r'^\s*$', np.nan, regex=True).dropna()

# Shuffle entire data before splitting
original_df = original_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into train and val
train_size = int(0.8 * len(original_df))
train_df = original_df.iloc[:train_size].reset_index(drop=True)
val_df = original_df.iloc[train_size:].reset_index(drop=True)

# ========== Oversample Train Only ========== #
class_counts = train_df['category'].value_counts()
max_count = class_counts.max()
balanced_train_df = pd.DataFrame()

for cat in class_counts.index:
    subset = train_df[train_df['category'] == cat]
    repeat_factor = int(np.ceil(max_count / len(subset)))
    oversampled = pd.concat([subset] * repeat_factor, ignore_index=True)
    balanced_train_df = pd.concat([balanced_train_df, oversampled], ignore_index=True)

train_df = balanced_train_df.sample(frac=1, random_state=42).reset_index(drop=True)

# ========== Label Encoding ========== #
label_encoder = LabelEncoder()
for d in [train_df, val_df]:
    d['label_encoded'] = label_encoder.fit_transform(d['category'])
    d['label_categorical'] = d['label_encoded'].apply(lambda x: to_categorical(x, num_classes=NUM_CLASSES))

print("\n✅ Label mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# ========== Find Majority Class ========== #
majority_class = train_df['label_encoded'].value_counts().idxmax()

# ========== Data Generator ========== #
class TifDataGenerator(Sequence):
    def __init__(self, df, image_folder, batch_size, img_size, channels, augment=False):
        self.df = df.reset_index(drop=True)
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.img_size = img_size
        self.channels = channels
        self.augment = augment
        self.flip_count = 0
        self.brightness_count = 0

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        X, y = [], []
        batch_indexes = range(index * self.batch_size, min((index + 1) * self.batch_size, len(self.df)))

        for i in batch_indexes:
            image_path = os.path.join(self.image_folder, self.df.iloc[i]['tif_path'])
            label = self.df.iloc[i]['label_encoded']

            try:
                with rasterio.open(image_path) as src:
                    img = src.read().astype(np.float32)

                    # Safely normalize each band
                    for b in range(img.shape[0]):
                        band = img[b]
                        band_min = np.nanmin(band)
                        band_max = np.nanmax(band)
                        if np.isnan(band_min) or np.isnan(band_max) or (band_max - band_min) < 1e-6:
                            img[b] = np.zeros_like(band)
                        else:
                            img[b] = (band - band_min) / (band_max - band_min)

                    img = np.transpose(img, (1, 2, 0))
                    
                    # If more channels than needed
                    if img.shape[2] >= self.channels:
                        img = img[:, :, :self.channels]

                    img_resized = tf.image.resize(img, self.img_size).numpy()

                    # Remove NaNs and infs after resizing
                    img_resized = np.nan_to_num(img_resized, nan=0.0, posinf=0.0, neginf=0.0)

                    if self.augment and label != majority_class:
                        if tf.random.uniform(()) > 0.5:
                            img_resized = tf.image.random_flip_left_right(img_resized)
                            self.flip_count += 1
                        if tf.random.uniform(()) > 0.5:
                            img_resized = tf.image.random_brightness(img_resized, max_delta=0.05)
                            self.brightness_count += 1

                    X.append(img_resized)
                    y.append(self.df.iloc[i]['label_categorical'])

            except Exception as e:
                print(f"❌ Skipping {image_path}: {e}")
                continue

        # ✅ Skip empty batches to avoid model input errors
        if len(X) == 0:
            return self.__getitem__((index + 1) % self.__len__())

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ========== Generators ========== #
train_gen = TifDataGenerator(train_df, IMAGE_FOLDER, BATCH_SIZE, IMG_SIZE, CHANNELS, augment=True)
val_gen = TifDataGenerator(val_df, IMAGE_FOLDER, BATCH_SIZE, IMG_SIZE, CHANNELS, augment=False)

# ========== Model Definition ========== #
with tf.device('/GPU:0'):
    model = Sequential([
        Conv2D(32, (7, 7), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], CHANNELS)),
        MaxPooling2D(2, 2),
        Conv2D(64, (5, 5), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=[early_stop])

# ========== Save Model ========== #
model.save("crop_health_generator_model6.h5")
print("\n✅ Model saved as crop_health_generator_model6.h5")

# ========== Plotting ========== #
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# ========== Confusion Matrix ========== #
y_true, y_pred = [], []
for X_batch, y_batch in val_gen:
    preds = model.predict(X_batch)
    y_true.extend(np.argmax(y_batch, axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)

plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix (Validation Set)")
plt.show()

# ========== Show Augmentation Counts ========== #
print(f"\n🌀 Total flips applied: {train_gen.flip_count}")
print(f"💡 Total brightness changes applied: {train_gen.brightness_count}")
