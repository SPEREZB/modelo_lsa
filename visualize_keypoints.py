import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob

def load_keypoints(json_file):
    """Carga los keypoints desde un archivo JSON"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def plot_hand_3d(keypoints, frame_idx=0, show_connections=True):
    """
    Dibuja los keypoints de la mano en 3D
    
    Args:
        keypoints: Lista de keypoints [x1,y1,z1, x2,y2,z2, ...]
        frame_idx: Índice del frame a visualizar (por defecto: 0)
        show_connections: Si es True, muestra las conexiones entre puntos
    """
    # Obtener los keypoints del frame seleccionado
    if isinstance(keypoints[0][0], list):  # Si hay múltiples frames
        frame_keypoints = keypoints[frame_idx]
    else:  # Si solo hay un frame
        frame_keypoints = keypoints
    
    # Convertir a numpy array y darle forma (21, 3)
    points = np.array(frame_keypoints).reshape(-1, 3)
    
    # Crear la figura 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extraer coordenadas x, y, z
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    # Dibujar solo los puntos azules
    ax.scatter(x, y, z, c='blue', s=50)
    
    # Definir conexiones de los dedos (índices basados en MediaPipe Hands)
    connections = [
        # Palm
        [0, 1], [1, 5], [5, 9], [9, 13], [13, 17], [0, 17],
        # Thumb
        [0, 1], [1, 2], [2, 3], [3, 4],
        # Index
        [0, 5], [5, 6], [6, 7], [7, 8],
        # Middle
        [9, 10], [10, 11], [11, 12],
        # Ring
        [13, 14], [14, 15], [15, 16],
        # Pinky
        [0, 17], [17, 18], [18, 19], [19, 20]
    ]
    
    if show_connections:
        for connection in connections:
            start = points[connection[0]]
            end = points[connection[1]]
            ax.plot([start[0], end[0]], 
                    [start[1], end[1]], 
                    [start[2], end[2]], 'k-')
    
    # Configurar etiquetas y título
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Visualización 3D de Keypoints (Frame {frame_idx + 1})')
    
    # Ajustar la vista para mejor visualización
    ax.view_init(elev=20, azim=30)
    
    # Ajustar los límites para que la mano se vea centrada
    max_range = max(x.max()-x.min(), y.max()-y.min(), z.max()-z.min()) * 0.5
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.show()

def list_json_files():
    """Lista todos los archivos JSON en el directorio data/keypoints_json"""
    json_dir = os.path.join('data', 'keypoints_json')
    if not os.path.exists(json_dir):
        print(f"Error: El directorio {json_dir} no existe.")
        return []
    
    json_files = glob.glob(os.path.join(json_dir, '*.json'))
    if not json_files:
        print(f"No se encontraron archivos JSON en {json_dir}")
    return json_files

def select_json_file():
    """Permite al usuario seleccionar un archivo JSON de la lista"""
    json_files = list_json_files()
    if not json_files:
        return None
    
    print("\nArchivos JSON encontrados:")
    for i, file in enumerate(json_files, 1):
        print(f"{i}. {os.path.basename(file)}")
    
    while True:
        try:
            selection = input("\nSeleccione el número del archivo a visualizar (o 'q' para salir): ")
            if selection.lower() == 'q':
                return None
            
            selection = int(selection) - 1
            if 0 <= selection < len(json_files):
                return json_files[selection]
            print("Número fuera de rango. Intente de nuevo.")
        except ValueError:
            print("Por favor ingrese un número válido.")

def main():
    # Obtener la ruta del archivo seleccionado
    json_file = select_json_file()
    if not json_file:
        print("No se seleccionó ningún archivo. Saliendo...")
        return
    
    print(f"\nCargando: {json_file}")
    
    # Cargar los keypoints
    try:
        data = load_keypoints(json_file)
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return
    
    # Verificar si hay metadatos
    if 'metadata' in data:
        print(f"\n--- Metadatos ---")
        print(f"Etiqueta: {data['metadata'].get('label', 'No especificada')}")
        print(f"Número de frames: {data['metadata'].get('frames', len(data['keypoints']))}")
        print(f"Timestamp: {data['metadata'].get('timestamp', 'No especificado')}")
    
    # Preguntar por el frame a visualizar
    num_frames = len(data['keypoints'])
    frame_idx = 0
    if num_frames > 1:
        while True:
            try:
                frame_input = input(f"\nIngrese el número de frame a visualizar (0-{num_frames-1}, o 'todos' para animación): ")
                if frame_input.lower() == 'todos':
                    frame_idx = 'all'
                    break
                frame_idx = int(frame_input)
                if 0 <= frame_idx < num_frames:
                    break
                print(f"Por favor ingrese un número entre 0 y {num_frames-1}")
            except ValueError:
                print("Entrada inválida. Intente de nuevo.")
    
    # Visualizar los keypoints
    if frame_idx == 'all':
        print("\nMostrando animación de todos los frames. Cierre la ventana para terminar.")
        for i in range(num_frames):
            plot_hand_3d(data['keypoints'], frame_idx=i)
            plt.pause(0.1)  # Pausa corta entre frames
            if not plt.get_fignums():
                break  # Salir si se cierra la ventana
    else:
        plot_hand_3d(data['keypoints'], frame_idx=frame_idx)
    
    plt.show()

if __name__ == "__main__":
    main()
