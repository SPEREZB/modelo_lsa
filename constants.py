import os
import cv2

# SETTINGS
MIN_LENGTH_FRAMES = 5
LENGTH_KEYPOINTS = 126
MODEL_FRAMES = 15

# PATHS
ROOT_PATH = os.getcwd()
FRAME_ACTIONS_PATH = os.path.join(ROOT_PATH, "frame_actions")
DATA_PATH = os.path.join(ROOT_PATH, "data")
DATA_JSON_PATH = os.path.join(DATA_PATH, "data.json")
MODEL_FOLDER_PATH = os.path.join(ROOT_PATH, "models")
MODEL_PATH = os.path.join(MODEL_FOLDER_PATH, f"letters_model.keras")
KEYPOINTS_PATH = os.path.join(DATA_PATH, "keypoints")  # Para archivos binarios .h5
KEYPOINTS_JSON_PATH = os.path.join(DATA_PATH, "keypoints_json")  # Para archivos JSON
WORDS_JSON_PATH = os.path.join(MODEL_FOLDER_PATH, "words.json")

# SHOW IMAGE PARAMETERS
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SIZE = 1.5
FONT_POS = (5, 30)

class WordsConfig:
    """Clase centralizada para las palabras del modelo LSP"""
    # WORDS = [
    #     "adios", "aprender", "bien", "chau", "cocinar", "comer", "comoestas",
    #     "dormir", "el", "entender", "estudiar", "gracias", "hastamañana", "hola",
    #     "informe", "investigar", "leer", "legusta", "mal", "mellamo", "mirame",
    #     "muymal", "nopoder", "nolegusta", "nosabia", "nosvemos", "perder", "perdon",
    #     "problema", "saber", "seacabo", "tienesrazon", "timido", "unpoco", "yo"
    # ]
    WORDS = [
        "adios", "aprender", "bien", "chau", "cocinar", "comer", "comoestas",
        "dormir", "el", "estudiar", "gracias", "hola", "informe", "investigar",
        "leer", "legusta", "mirame", "nolegusta", "perder", "perdon",
        "tienesrazon", "timido", "yo"
    ]
    
    LETTERS = list("abcdefklmnopqrsuvxz")
    
    @classmethod
    def get_classes(cls):
        """Retorna las clases combinadas (letras + palabras) ordenadas alfabéticamente"""
        return sorted(cls.LETTERS + cls.WORDS, key=lambda x: x.lower())


words_text = {
    "adios": "ADIÓS",
    "bien": "BIEN",
    "buenas_noches": "BUENAS NOCHES",
    "buenas_tardes": "BUENAS TARDES",
    "buenos_dias": "BUENOS DÍAS",
    "como_estas": "COMO ESTÁS",
    "disculpa": "DISCULPA",
    "gracias": "GRACIAS",
    "hola": "HOLA",
    "mal": "MAL",
    "mas_o_menos": "MAS O MENOS",
    "me_ayudas": "ME AYUDAS",
    "por_favor": "POR FAVOR",
}