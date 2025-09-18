from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.regularizers import l2
from constants import LENGTH_KEYPOINTS

def get_model(sequence_length, num_classes, features):
    """
    Crea un modelo LSTM para clasificación de secuencias.
    """
    model = Sequential()
    
    model.add(Masking(mask_value=0., input_shape=(sequence_length, features)))
    
    # Primera capa LSTM
    model.add(LSTM(
        128,
        return_sequences=True
    ))
    model.add(Dropout(0.5))
    
    # Segunda capa LSTM
    model.add(LSTM(
        256,
        return_sequences=False,
        kernel_regularizer=l2(0.001)
    ))
    model.add(Dropout(0.5))
    
    # Capas densas para clasificación
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    
    # Capa de salida
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compilar el modelo
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

