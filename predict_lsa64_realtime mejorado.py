import cv2
import numpy as np
import mediapipe as mp
import json
import time
from tensorflow.keras.models import load_model

# ===============================
# CONFIG
# ===============================

MODEL_PATH = "models/letters_model.keras"
WORDS_JSON = "models/words.json"

SEQUENCE_LENGTH = 107
NUM_FEATURES = 126

MOTION_THRESHOLD = 0.004
CONFIDENCE_THRESHOLD = 0.40
COOLDOWN_TIME = 1.2  # segundos

# ===============================
# LABELS (MISMO ORDEN)
# ===============================

with open(WORDS_JSON, "r", encoding="utf-8") as f:
    LABELS = json.load(f)["word_ids"]

LABELS = [w.lower() for w in LABELS]

print("✔ Labels cargados:", LABELS)

# ===============================
# MEDIAPIPE
# ===============================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ===============================
# MODELO
# ===============================

model = load_model(MODEL_PATH)
print("✅ Modelo cargado")

# ===============================
# FUNCIONES
# ===============================

def extract_keypoints(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    keypoints = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks[:2]:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

    if len(keypoints) < NUM_FEATURES:
        keypoints.extend([0.0] * (NUM_FEATURES - len(keypoints)))

    return np.array(keypoints, dtype=np.float32)


def motion_level(a, b):
    return np.mean(np.abs(a - b))


# ===============================
# LOOP PRINCIPAL
# ===============================

cap = cv2.VideoCapture(0)

state = "WAITING"
sequence = []
prev_keypoints = None
last_prediction = ""
cooldown_start = 0
conf = 0

print("\n🎥 Sistema activo — detección automática\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    keypoints = extract_keypoints(frame)

    motion = 0
    if prev_keypoints is not None:
        motion = motion_level(prev_keypoints, keypoints)
    prev_keypoints = keypoints

    now = time.time()

    # ===============================
    # WAITING
    # ===============================
    if state == "WAITING":
        cv2.putText(frame, "Esperando gesto...",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (200, 200, 0), 2)

        if motion > MOTION_THRESHOLD:
            sequence = []
            state = "RECORDING"

    # ===============================
    # RECORDING
    # ===============================
    elif state == "RECORDING":
        sequence.append(keypoints)

        cv2.putText(frame,
            f"Grabando {len(sequence)}/{SEQUENCE_LENGTH}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 255), 2)

        if len(sequence) >= SEQUENCE_LENGTH:
            state = "PREDICTING"

    # ===============================
    # PREDICTING
    # ===============================
    elif state == "PREDICTING":
        seq = np.array(sequence)

        if seq.shape != (SEQUENCE_LENGTH, NUM_FEATURES):
            state = "WAITING"
            continue

        seq = np.expand_dims(seq, axis=0)

        preds = model.predict(seq, verbose=0)[0]
        idx = np.argmax(preds)
        conf = preds[idx]

        last_prediction = f"{LABELS[idx]} ({conf:.2f})"

        print("\n🧠 RESULTADO:")
        for i, p in enumerate(preds):
            print(f"{LABELS[i]:12s}: {p:.3f}")

        cooldown_start = now
        state = "COOLDOWN"

    # ===============================
    # COOLDOWN
    # ===============================
    elif state == "COOLDOWN":
        color = (0, 255, 0) if conf >= CONFIDENCE_THRESHOLD else (0, 0, 255)

        cv2.putText(frame,
            f"Detectado: {last_prediction}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
            1, color, 2)

        if now - cooldown_start > COOLDOWN_TIME:
            state = "WAITING"

    cv2.imshow("Reconocimiento LSA", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
