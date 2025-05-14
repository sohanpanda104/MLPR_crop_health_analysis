import os
import warnings
import logging
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.image import resize
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# print("hello")

# ─── CONFIG ───────────────────────────────────────────────────────────────
CSV_PATH = "../combined_dataset/16_bands/output2.csv"
IMAGE_FOLDER = "../combined_dataset/16_bands/downloads"
IMG_SIZE = (128, 128)
CHANNELS = 16
BATCH_SIZE = 8
NUM_CLASSES = 4
EPOCHS = 15
PATIENCE = 5
EXPECTED_LABEL_COL = 'category'

# ─── Suppress Warnings ─────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger('rasterio').setLevel(logging.ERROR)

# ─── Load & Preprocess ─────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df.dropna(subset=[EXPECTED_LABEL_COL, 'tif_path'], inplace=True)
df = df[df['tif_path'].str.lower().str.endswith('.tif')].reset_index(drop=True)

# Duration calculation
df['SDate'] = pd.to_datetime(df['SDate'])
df['HDate'] = pd.to_datetime(df['HDate'])
df['DurationDays'] = (df['HDate'] - df['SDate']).dt.days

# ─── Train/Val Split & Oversample ──────────────────────────────────────────
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
split = int(0.8 * len(df))
train_df_raw, val_df = df.iloc[:split], df.iloc[split:]

class_counts = train_df_raw[EXPECTED_LABEL_COL].value_counts()
max_count = class_counts.max()
oversampled = []
for cat, cnt in class_counts.items():
    subset = train_df_raw[train_df_raw[EXPECTED_LABEL_COL] == cat]
    reps = int(np.ceil(max_count / cnt))
    oversampled.append(pd.concat([subset]*reps, ignore_index=True))
train_df = pd.concat(oversampled, ignore_index=True).sample(frac=1, random_state=42)
majority_class = train_df[EXPECTED_LABEL_COL].value_counts().idxmax()

# ─── Label Encoding ────────────────────────────────────────────────────────
le = LabelEncoder()
y_train = le.fit_transform(train_df[EXPECTED_LABEL_COL])
y_val = le.transform(val_df[EXPECTED_LABEL_COL])
y_train_cat = to_categorical(y_train, NUM_CLASSES)
y_val_cat = to_categorical(y_val, NUM_CLASSES)

# ─── Feature Encoding ───────────────────────────────────────────────────────
num_cols = ['CropCoveredArea','CHeight','IrriCount','WaterCov','ExpYield','DurationDays',
            'AvgRainfall(mm)', 'AvgMinHumidity(%)','AvgMaxHumidity(%)','Min Temp Avg','Max Temp Avg']
cat_cols = ['Crop','District','Sub-District','CNext','CLast','CTransp','IrriType','IrriSource','Season']

scaler = StandardScaler().fit(train_df[num_cols])
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train_df[cat_cols])

X_tab_train = np.hstack([scaler.transform(train_df[num_cols]), ohe.transform(train_df[cat_cols])])
X_tab_val   = np.hstack([scaler.transform(val_df[num_cols]),   ohe.transform(val_df[cat_cols])])

# ─── Dual Data Generator ───────────────────────────────────────────────────
class DualDataGenerator(Sequence):
    def __init__(self, df, X_tab, y, img_folder, batch_size, img_size, channels, majority_label):
        self.df = df.reset_index(drop=True)
        self.X_tab = X_tab
        self.y = y
        self.img_folder = img_folder
        self.batch_size = batch_size
        self.img_size = img_size
        self.channels = channels
        self.maj_index = le.transform([majority_label])[0]

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx+1)*self.batch_size, len(self.df))
        batch = self.df.iloc[start:end]

        imgs, tabs, labs = [], [], []
        for i, row in batch.iterrows():
            path = os.path.join(self.img_folder, row['tif_path'])
            try:
                with rasterio.open(path) as src:
                    img = src.read().astype(np.float32)
                for b in range(img.shape[0]):
                    band = img[b]
                    img[b] = (band - band.min()) / (band.max() - band.min() + 1e-5)
                img = np.transpose(img[:self.channels], (1,2,0))
                img = resize(img, self.img_size).numpy()
                if le.transform([row[EXPECTED_LABEL_COL]])[0] != self.maj_index:
                    img = tf.image.random_flip_left_right(img)
                    img = tf.image.random_brightness(img, 0.05)
            except Exception as e:
                print(f"❌ Skipping {path}: {e}")
                continue

            imgs.append(img)
            tabs.append(self.X_tab[i])
            labs.append(self.y[i])

        return (np.array(imgs, dtype=np.float32), np.array(tabs, dtype=np.float32)), np.array(labs, dtype=np.float32)

train_gen = DualDataGenerator(train_df, X_tab_train, y_train_cat, IMAGE_FOLDER, BATCH_SIZE, IMG_SIZE, CHANNELS, majority_class)
val_gen   = DualDataGenerator(val_df,   X_tab_val,   y_val_cat,   IMAGE_FOLDER, BATCH_SIZE, IMG_SIZE, CHANNELS, majority_class)

# ─── Model Definition ──────────────────────────────────────────────────────
img_in = Input((*IMG_SIZE, CHANNELS), name="Image_Input")
x = Conv2D(32, (3,3), activation='relu')(img_in)
x = MaxPooling2D()(x)
x = Conv2D(64, (3,3), activation='relu')(x)
x = MaxPooling2D()(x)
x = Conv2D(128, (3,3), activation='relu')(x)
x = MaxPooling2D()(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)

tab_in = Input((X_tab_train.shape[1],), name="Tabular_Input")
y1 = Dense(64, activation='relu')(tab_in)
y1 = Dropout(0.4)(y1)
y1 = Dense(32, activation='relu')(y1)

merged = concatenate([x, y1])
z = Dense(128, activation='relu')(merged)
z = Dropout(0.4)(z)
out = Dense(NUM_CLASSES, activation='softmax')(z)

model = Model([img_in, tab_in], out)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ─── Training Loop ─────────────────────────────────────────────────────────
best_val_loss = np.inf
wait = 0
train_losses, train_accs = [], []
val_losses, val_accs = [], []

for epoch in range(1, EPOCHS + 1):
    tloss, tacc = 0.0, 0.0
    for step in range(len(train_gen)):
        (imgs, tabs), labs = train_gen[step]
        loss, acc = model.train_on_batch([imgs, tabs], labs)
        tloss += loss
        tacc  += acc
    tloss /= len(train_gen)
    tacc  /= len(train_gen)
    train_losses.append(tloss)
    train_accs.append(tacc)

    vloss, vacc = 0.0, 0.0
    for step in range(len(val_gen)):
        (imgs, tabs), labs = val_gen[step]
        loss, acc = model.test_on_batch([imgs, tabs], labs)
        vloss += loss
        vacc  += acc
    vloss /= len(val_gen)
    vacc  /= len(val_gen)
    val_losses.append(vloss)
    val_accs.append(vacc)

    print(f"Epoch {epoch}/{EPOCHS} — train_loss: {tloss:.4f}, train_acc: {tacc:.4f} | val_loss: {vloss:.4f}, val_acc: {vacc:.4f}")

    if vloss < best_val_loss:
        best_val_loss = vloss
        model.save("dual_crop_health_model_updated.h5")
        wait = 0
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"\u2192 Early stopping at epoch {epoch}")
            break

# ─── Plot Curves ───────────────────────────────────────────────────────────
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs,   label='Val Acc')
plt.title("Accuracy"); plt.xlabel("Epoch"); plt.legend()

plt.subplot(1,2,2)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses,   label='Val Loss')
plt.title("Loss"); plt.xlabel("Epoch"); plt.legend()

plt.tight_layout()
plt.show()

# dual model with temp and rainfall
