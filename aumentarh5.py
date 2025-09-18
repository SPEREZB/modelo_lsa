import os
import numpy as np
import h5py
from scipy import ndimage

def augment_sequence(sequence, num_augmentations=4):
    """Aplica transformaciones aleatorias a una secuencia"""
    augmented = [sequence]
    
    for _ in range(num_augmentations):
        # 1. Copiar la secuencia original
        new_seq = sequence.copy()
        
        # 2. Aplicar ruido gaussiano
        noise = np.random.normal(0, 0.01, new_seq.shape)
        new_seq += noise
        
        # 3. Aplicar rotación 3D
        angles = np.random.uniform(-10, 10, 3)  # Grados
        for i, angle in enumerate(angles):
            new_seq = ndimage.rotate(
                new_seq, 
                angle, 
                axes=((i+1)%3, (i+2)%3), 
                reshape=False, 
                mode='nearest'
            )
        
        # 4. Guardar la secuencia aumentada
        augmented.append(new_seq)
    
    return np.array(augmented)

# Ruta a la carpeta con tus archivos .h5
input_folder = 'C:/Users/Perez/Desktop/Desarrollo/Modelo_LSP/modelo_lstm_lsp/data/keypoints'
output_folder = 'C:/Users/Perez/Desktop/Desarrollo/Modelo_LSP/modelo_lstm_lsp/data/keypoints_augmented'
# Crear carpeta de salida si no existe
os.makedirs(output_folder, exist_ok=True)

# Procesar cada archivo .h5
for filename in os.listdir(input_folder):
    if filename.endswith('.h5'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f'augmented_{filename}')
        
        print(f'Procesando: {filename}')
        
        # Cargar datos
        with h5py.File(input_path, 'r') as f:
            X = f['X'][()]  # Ajusta según tu estructura
            y = f['y'][()]  # Ajusta según tu estructura
        
        # Aumentar datos
        X_augmented = []
        y_augmented = []
        
        for i in range(len(X)):
            augmented = augment_sequence(X[i])
            X_augmented.append(augmented)
            y_augmented.extend([y[i]] * 5)  # 1 original + 4 aumentadas
        
        X_augmented = np.concatenate(X_augmented, axis=0)
        y_augmented = np.array(y_augmented)
        
        # Guardar datos aumentados
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('X', data=X_augmented)
            f.create_dataset('y', data=y_augmented)
            
        print(f'Guardado: {output_path}')