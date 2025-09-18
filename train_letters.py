import os
import numpy as np
import h5py
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from model import get_model
from constants import *
from helpers import get_word_ids
import json 
SEQUENCE_LENGTH = 107
TOTAL_KEYPOINTS = 1530

def load_letter_sequences(data_dir='data/keypoints'):
    sequences = []
    labels = []
    word_ids = get_word_ids(WORDS_JSON_PATH)
    
    # Keypoints por mano: 21 puntos * 3 coordenadas (x, y, z)
    KEYPOINTS_PER_HAND = 63  # 21 * 3
    MAX_HANDS = 2  # Máximo de manos que vamos a procesar

  

    # Obtener la lista de archivos .h5
    h5_files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
    print(f"\nEncontrados {len(h5_files)} archivos .h5 en {data_dir}")

    for filename in h5_files:
        try:
            letter = filename.split('_')[0].lower()
            if letter in word_ids:
                filepath = os.path.join(data_dir, filename)
                
                with h5py.File(filepath, 'r') as f:
                    if 'keypoints' in f:
                        data = np.array(f['keypoints'], dtype=np.float32)  # (frames, n_keypoints)

                        # Padding/recorte frames
                        if data.shape[0] < SEQUENCE_LENGTH:
                            pad_frames = np.zeros((SEQUENCE_LENGTH - data.shape[0], data.shape[1]), dtype=np.float32)
                            data = np.vstack([data, pad_frames])
                        elif data.shape[0] > SEQUENCE_LENGTH:
                            data = data[:SEQUENCE_LENGTH]

                        # Padding/recorte keypoints
                        if data.shape[1] < TOTAL_KEYPOINTS:
                            pad_features = np.zeros((SEQUENCE_LENGTH, TOTAL_KEYPOINTS - data.shape[1]), dtype=np.float32)
                            data = np.hstack([data, pad_features])
                        elif data.shape[1] > TOTAL_KEYPOINTS:
                            data = data[:, :TOTAL_KEYPOINTS]

                        sequences.append(data)
                        labels.append(word_ids.index(letter))

                    
        except Exception as e:
            print(f"Error procesando {filename}: {str(e)}")
        
    return sequences, labels, word_ids


def prepare_data(sequences, labels, num_classes):
    """
    Prepara los datos para el entrenamiento
    """
    # Encontrar la longitud máxima de secuencia
    max_sequence_length = max(len(seq) for seq in sequences)
    print(f"\nLongitud máxima de secuencia: {max_sequence_length}")
    
    # Asegurar que todas las secuencias tengan la misma longitud
    X = np.array(sequences, dtype=np.float32)  # Forma: (num_samples, SEQUENCE_LENGTH, TOTAL_KEYPOINTS)
    y = to_categorical(labels, num_classes=num_classes)
    return X, y, SEQUENCE_LENGTH

    
    # Convertir a arrays de numpy
    X = np.array(padded_sequences)

    # Redimensionar a (num_samples, timesteps, features)
    timesteps = X.shape[1]
    features = X.shape[2] if X.ndim == 3 else X.shape[1] // timesteps
    X = X.reshape((X.shape[0], timesteps, -1))

    y = to_categorical(labels, num_classes=num_classes)
    
    return X, y, max_sequence_length

def train_letters_model(model_path, epochs=100, batch_size=16):
    """
    Entrena un modelo para reconocimiento de letras
    """
    # Cargar datos
    print("Cargando secuencias de letras...")
    sequences, labels, word_ids = load_letter_sequences()
    
    if not sequences:
        print("No se encontraron secuencias válidas para entrenar.")
        return
    
    print(f"\nTotal de secuencias cargadas: {len(sequences)}")
    print(f"Letras únicas encontradas: {len(set(labels))}")
    
    # Preparar datos
    X, y, sequence_length = prepare_data(sequences, labels, len(word_ids))
    
    # Dividir en conjuntos de entrenamiento y validación
    # Para pocas muestras, usamos validación cruzada o dejamos una muestra por clase para validación
    from sklearn.model_selection import StratifiedShuffleSplit
    
    # Si hay muy pocas muestras, mejor usar validación cruzada
    if len(labels) < 10:  # Si hay menos de 10 muestras
        print("Usando validación cruzada debido a pocas muestras")
        # Devolvemos todo para entrenamiento y usaremos validación cruzada
        X_train, y_train = X, y
        X_val, y_val = X[:0], y[:0]  # Arrays vacíos para mantener la estructura
    else:
        # Si hay suficientes muestras, hacemos la división normal
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(split.split(X, y))
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
    
    print(f"\nDatos de entrenamiento: {len(X_train)} muestras")
    print(f"Datos de validación: {len(X_val)} muestras")
    
    print(f"\nDatos de entrenamiento: {X_train.shape}")
    print(f"Datos de validación: {X_val.shape}")
    
    # Crear y compilar el modelo
    print("\nCreando modelo...")
    features = X_train.shape[2]  # número de keypoints por frame
    model = get_model(sequence_length, len(word_ids), features)
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Entrenar el modelo
    print("\nEntrenando modelo...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    
    # Guardar el modelo final
    model.save(model_path)
    metadata = {
    'max_sequence_length': sequence_length,
    'num_classes': len(word_ids)
    }
    with open(MODEL_LETTERS_PATH.replace('.keras', '_metadata.json'), 'w') as f:
        json.dump(metadata, f)
    print(f"\nModelo guardado en: {model_path}")
    
    return history

if __name__ == "__main__":
    # Ruta donde se guardará el modelo entrenado
    MODEL_LETTERS_PATH = os.path.join(MODEL_FOLDER_PATH, "letters_model.keras")
    
    # Entrenar el modelo
    train_letters_model(
        model_path=MODEL_LETTERS_PATH,
        epochs=200,
        batch_size=16
    )
