import os
import json
import h5py
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

# ===============================
# CONFIG
# ===============================
DATA_DIR = "data/keypoints"
WORDS_JSON = "models/words.json"
MODEL_PATH = "models/letters_model.keras"

SEQUENCE_LENGTH = 107
FEATURES = 126
EPOCHS = 200
BATCH_SIZE = 16

# ===============================
# LIMPIEZA DE NOMBRE
# ===============================
def clean_word(filename):
    name = filename.lower()
    name = name.replace(".h5", "")
    name = name.rstrip("0123456789")
    name = name.rstrip("_")
    return name

# ===============================
# CARGAR WORDS.JSON (CORRECTO)
# ===============================
with open(WORDS_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

words = data["word_ids"]   # 🔥 AQUÍ ESTABA EL ERROR
words = [w.lower() for w in words]

word_to_id = {w: i for i, w in enumerate(words)}

print("✔ Palabras:", len(words))

# ===============================
# CARGAR DATASET
# ===============================
def load_dataset():
    X, y = [], []

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".h5")]

    for file in files:
        word = clean_word(file)

        if word not in word_to_id:
            print(f"⚠️ No existe en words.json → {word}")
            continue

        label = word_to_id[word]

        with h5py.File(os.path.join(DATA_DIR, file), "r") as f:
            seq = np.array(f["keypoints"], dtype=np.float32)

            # padding frames
            if seq.shape[0] > SEQUENCE_LENGTH:
                seq = seq[:SEQUENCE_LENGTH]
            else:
                pad = np.zeros((SEQUENCE_LENGTH - seq.shape[0], FEATURES))
                seq = np.vstack([seq, pad])

            X.append(seq)
            y.append(label)

    return np.array(X), np.array(y)

# ===============================
# ENTRENAMIENTO
# ===============================
print("\n📥 Cargando dataset...")
X, y = load_dataset()

print("✔ Samples:", X.shape)

if len(X) == 0:
    print("\n❌ ERROR: no se cargó ninguna muestra")
    exit()

y_cat = to_categorical(y, num_classes=len(words))

X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ===============================
# MODELO
# ===============================
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, FEATURES)),
    BatchNormalization(),
    Dropout(0.3),

    LSTM(64),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation="relu"),
    Dropout(0.3),

    Dense(len(words), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks = [
    EarlyStopping(patience=25, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, save_best_only=True)
]

print("\n🚀 ENTRENANDO MODELO...\n")

model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

model.save(MODEL_PATH)

print("\n✅ MODELO ENTRENADO CORRECTAMENTE")
print("📁", MODEL_PATH)
