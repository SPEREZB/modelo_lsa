import cv2
import requests
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json

def draw_landmarks_on_image(rgb_image, detection_result):
    """Dibuja los landmarks de las manos en la imagen """
    annotated_image = np.copy(rgb_image)

    if detection_result.multi_hand_landmarks:
        for hand_landmarks in detection_result.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style())
    
    return annotated_image


def send_video(api_url, delay=0.033):  
    # Inicializar MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    
    # Inicializar cámara con resolución 720p
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    last_prediction = ""
    last_confidence = 0.0
    last_hand_detected = False
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Error al leer el frame")
                break
                
            # Voltear la imagen horizontalmente para una visualización tipo selfie
            frame = cv2.flip(frame, 1)
             
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
             
            results = hands.process(rgb_frame)
            
            # Dibujar landmarks si se detectan manos
            annotated_image = frame.copy()
            if results.multi_hand_landmarks:
                annotated_image = draw_landmarks_on_image(annotated_image, results)
                hand_detected = True
            else:
                hand_detected = False
            
            # Solo enviar a la API si se detectan manos
            if hand_detected and (not last_hand_detected or frame_count % 2 == 0):
                # Reducir tamaño para la API (480p)
                small_frame = cv2.resize(frame, (640, 480))
                _, img_encoded = cv2.imencode('.jpg', small_frame, 
                                           [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                
                try:
                    # Enviar frame a la API
                    response = requests.post(
                        api_url,
                        files={'image': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')},
                        timeout=0.5  
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if 'prediction' in result and 'confidence' in result:
                            last_prediction = result['prediction']
                            last_confidence = result['confidence']
                
                except requests.exceptions.RequestException:
                    pass  # Ignorar timeouts breves
            
            last_hand_detected = hand_detected
            frame_count += 1
            
            # Mostrar información en el frame
            info_text = f"{last_prediction} ({last_confidence:.2f})" if last_prediction else ""
            if info_text:
                cv2.putText(annotated_image, info_text, 
                          (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                          1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Mostrar FPS
            fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.putText(annotated_image, f'FPS: {fps:.1f}', 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Mostrar estado de detección
            status = "Mano detectada" if hand_detected else "Buscando manos..."
            cv2.putText(annotated_image, status, 
                       (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0) if hand_detected else (0, 0, 255), 2, cv2.LINE_AA)
            
            # Mostrar frame
            cv2.imshow('LSP Detector', annotated_image)
            
            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nDeteniendo...")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Iniciando detección de LSP...")
    print("Presiona 'q' para salir")
    send_video("http://127.0.0.1:5000/api/predict", delay=0.1)