import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QComboBox, QLineEdit, QWidget)
from PyQt6.QtCore import QTimer, Qt, QProcess
from PyQt6.QtGui import QImage, QPixmap
from constants import KEYPOINTS_PATH, DATA_PATH
import mediapipe as mp
import h5py
import os

class CaptureApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Captura de Letras LSA")
        self.setFixedSize(1000, 700)
        
        # Inicializar MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Variables
        self.cap = None
        self.recording = False
        self.frames = []
        self.letter = ""
        self.process = None
        
        # Configurar UI
        self.init_ui()
        
    def init_ui(self):
        # Widget principal
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Panel izquierdo - Vista de cámara
        self.camera_view = QLabel()
        self.camera_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_view.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        
        # Panel derecho - Controles
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # Título
        title = QLabel("Captura de Letras")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 20px;")
        
        # Selector de modo
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Solo manos", "Manos y rostro"])
        self.mode_combo.setStyleSheet("padding: 8px;")
        
        # Entrada de letra
        self.letter_input = QLineEdit()
        self.letter_input.setPlaceholderText("Ingrese la letra (a-z)")
        self.letter_input.setStyleSheet("padding: 8px;")
        
        # Botón de captura
        self.capture_btn = QPushButton("Iniciar Captura")
        self.capture_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.capture_btn.clicked.connect(self.toggle_capture)
        
        # Botones adicionales
        self.btn_train = QPushButton("Entrenar Modelo")
        self.btn_predict = QPushButton("Predecir en Tiempo Real")
        
        # Estilo para botones adicionales
        button_style = """
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px;
                margin: 5px 0;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """
        
        self.btn_train.setStyleSheet(button_style)
        self.btn_predict.setStyleSheet(button_style)
        
        # Conectar botones
        self.btn_train.clicked.connect(self.run_train)
        self.btn_predict.clicked.connect(self.run_predict)
        
        # Estado
        self.status_label = QLabel("Listo")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Agregar widgets al layout de controles
        control_layout.addWidget(title)
        control_layout.addWidget(QLabel("Modo de detección:"))
        control_layout.addWidget(self.mode_combo)
        control_layout.addSpacing(20)
        control_layout.addWidget(QLabel("Letra a capturar:"))
        control_layout.addWidget(self.letter_input)
        control_layout.addSpacing(20)
        control_layout.addWidget(self.capture_btn)
        control_layout.addSpacing(30)
        control_layout.addWidget(QLabel("Otras acciones:"))
        control_layout.addWidget(self.btn_train)
        control_layout.addWidget(self.btn_predict)
        control_layout.addStretch()
        control_layout.addWidget(self.status_label)
        
        # Agregar paneles al layout principal
        main_layout.addWidget(self.camera_view, 3)
        main_layout.addWidget(control_panel, 1)
        
        # Iniciar cámara
        self.start_camera()
        
    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~30 FPS
        
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Voltear la imagen horizontalmente
            frame = cv2.flip(frame, 1)
            
            # Procesar con MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            # Dibujar landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Mostrar frame
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            convert_to_qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
            p = convert_to_qt_format.scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio)
            self.camera_view.setPixmap(QPixmap.fromImage(p))
            
    def toggle_capture(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
            
    def start_recording(self):
        self.letter = self.letter_input.text().strip().lower()
        if not self.letter.isalpha() or len(self.letter) != 1:
            self.status_label.setText("Ingrese una letra válida (a-z)")
            return
            
        self.recording = True
        self.frames = []
        self.capture_btn.setText("Detener Captura")
        self.status_label.setText(f"Grabando letra: {self.letter}")
        
    def stop_recording(self):
        self.recording = False
        self.capture_btn.setText("Iniciar Captura")
        self.save_recording()
        
    def save_recording(self):
        if not self.frames:
            self.status_label.setText("Error: No se capturaron frames")
            return
            
        # Crear directorios si no existen
        os.makedirs(KEYPOINTS_PATH, exist_ok=True)
        os.makedirs(os.path.join(DATA_PATH, "videos"), exist_ok=True)
        
        # Guardar keypoints
        keypoints_array = np.array(self.frames)
        sample_num = len([f for f in os.listdir(KEYPOINTS_PATH) 
                         if f.startswith(f"{self.letter}_")]) + 1
        
        output_file = os.path.join(KEYPOINTS_PATH, f"{self.letter}_{sample_num}.h5")
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('keypoints', data=keypoints_array)
            
        self.status_label.setText(f"Guardado: {self.letter} (muestra {sample_num})")
    
    def run_train(self):
        try:
            self.process = QProcess()
            self.process.start('python', ['train_letters.py'])
            self.status_label.setText("Entrenamiento iniciado...")
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")

    def run_predict(self):
        try:
            self.process = QProcess()
            self.process.start('python', ['predict_letters_sliding.py'])
            self.status_label.setText("Predicción iniciada...")
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")

    def closeEvent(self, event):
        if hasattr(self, 'process') and self.process:
            self.process.terminate()
            self.process.waitForFinished()
        if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'hands'):
            self.hands.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CaptureApp()
    window.show()
    sys.exit(app.exec())