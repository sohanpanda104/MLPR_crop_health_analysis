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
import tensorflow as tf
import matplotlib.pyplot as plt

# ========== CONFIG ========== #
CSV_PATH = "../combined_dataset/16_bands/output.csv"
IMAGE_FOLDER = "../combined_dataset/16_bands/"
IMG_SIZE = (128, 128)
NUM_CLASSES = 4
BATCH_SIZE = 8
CHANNELS = 16

# ========== Load & Split ========== #
original_df = pd.read_csv(CSV_PATH)
original_df = original_df.replace(r'^\s*$', np.nan, regex=True).dropna()

# Shuffle and split before oversampling
df = original_df.sample(frac=1, random_state=42).reset_index(drop=True)
train_size = int(0.8 * len(df))
train_df_raw = df.iloc[:train_size].reset_index(drop=True)
val_df = df.iloc[train_size:].reset_index(drop=True)

# ========== Oversample Only Training Set ========== #
class_counts = train_df_raw['category'].value_counts()
max_count = class_counts.max()
balanced_train_df = pd.DataFrame()

for cat in class_counts.index:
    subset = train_df_raw[train_df_raw['category'] == cat]
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

# ========== Find majority class for augmentation condition ========== #
majority_class = train_df['label_encoded'].value_counts().idxmax()

# ========== Data Generator ========== #
class TifDataGenerator(Sequence):
    def __init__(self, df, image_folder, batch_size, img_size, channels):
        self.df = df.reset_index(drop=True)
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.img_size = img_size
        self.channels = channels

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
                    for b in range(img.shape[0]):
                        band = img[b]
                        img[b] = (band - band.min()) / (band.max() - band.min() + 1e-5)
                    img = np.transpose(img, (1, 2, 0))
                    if img.shape[2] >= self.channels:
                        img = img[:, :, :self.channels]
                    img_resized = resize(img, self.img_size).numpy()

                    if label != majority_class:
                        img_resized = tf.image.random_flip_left_right(img_resized)
                        img_resized = tf.image.random_brightness(img_resized, max_delta=0.05)

                    X.append(img_resized)
                    y.append(self.df.iloc[i]['label_categorical'])

            except Exception as e:
                print(f"❌ Skipping {image_path}: {e}")
                continue

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ========== Generators ========== #
train_gen = TifDataGenerator(train_df, IMAGE_FOLDER, BATCH_SIZE, IMG_SIZE, CHANNELS)
val_gen = TifDataGenerator(val_df, IMAGE_FOLDER, BATCH_SIZE, IMG_SIZE, CHANNELS)

# ========== Model Definition ========== #
with tf.device('/GPU:0'):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], CHANNELS)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
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

    # ========== Training ========== #
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        callbacks=[early_stop]
    )

# ========== Save Model ========== #
model.save("crop_health_generator_model8_3.h5")
print("\n✅ Model saved as crop_health_generator_model8_3.h5")

# ========== Plotting ========== #
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# no oversampling with 16 bands being used now. 
