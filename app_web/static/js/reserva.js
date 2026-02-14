// WebSocket client for LSP application
document.addEventListener('DOMContentLoaded', () => {
    // Connect to WebSocket server
    const socket = io();
    const statusDiv = document.getElementById('status');
    const resultDiv = document.getElementById('result');
    const confidenceDiv = document.getElementById('confidence');
    
    // Variable para evitar repeticiones rápidas
    let lastSpokenPrediction = '';
    let lastSpokenTime = 0;
    const MIN_TIME_BETWEEN_SPEECH_MS = 3000; // 2 segundos entre mensajes
    
    // Connection status
    socket.on('connect', () => {
        console.log('Conectado al servidor WebSocket');
        statusDiv.textContent = 'Conectado al servidor de reconocimiento';
    });
    
    socket.on('disconnect', () => {
        console.log('Desconectado del servidor WebSocket');
        statusDiv.textContent = 'Desconectado del servidor';
    });
    
    socket.on('connect_error', (error) => {
        console.error('Error de conexión WebSocket:', error);
        statusDiv.textContent = 'Error de conexión con el servidor';
    });
    
    // Handle prediction responses
    // Función para leer texto en voz alta
    function speakText(text) {
        // Verificar si la API de síntesis de voz está disponible
        if ('speechSynthesis' in window) {
            // Crear un nuevo objeto de síntesis de voz
            const utterance = new SpeechSynthesisUtterance(text);
            
            // Configurar opciones de voz
            utterance.lang = 'es-ES'; // Idioma español
            utterance.rate = 0.9; // Velocidad de habla (0.1 a 10)
            utterance.pitch = 1.0; // Tono de voz (0 a 2)
            
            // Obtener voces disponibles y seleccionar una en español si está disponible
            const voices = window.speechSynthesis.getVoices();
            const spanishVoice = voices.find(voice => voice.lang.startsWith('es'));
            if (spanishVoice) {
                utterance.voice = spanishVoice;
            }
            
            // Reproducir el mensaje
            window.speechSynthesis.speak(utterance);
        } else {
            console.warn('La síntesis de voz no está soportada en este navegador');
        }
    }
    
    socket.on('prediction', (data) => {
        if (data.error) {
            console.error('Error del servidor:', data.error);
            statusDiv.textContent = `Error: ${data.error}`;
            return;
        }
        
        if (data.success) {
            const currentPrediction = data.prediction;  // Solo el texto de la predicción
            const predictionText = `Letra: ${currentPrediction}`;
            resultDiv.textContent = predictionText;
            confidenceDiv.textContent = `Confianza: ${(data.confidence * 100).toFixed(2)}%`;
            
            // Verificar si la predicción es nueva y ha pasado el tiempo mínimo
            const currentTime = Date.now();
            const isNewPrediction = currentPrediction !== lastSpokenPrediction;
            const hasEnoughTimePassed = (currentTime - lastSpokenTime) > MIN_TIME_BETWEEN_SPEECH_MS;
            
            if (isNewPrediction && hasEnoughTimePassed) {
                speakText(currentPrediction);
                lastSpokenPrediction = currentPrediction;  // Solo guardamos la predicción, sin el prefijo
                lastSpokenTime = currentTime;
            }
        }
    });
    
    // Function to send frame to server
    function sendFrame(canvas) {
        if (!socket.connected) return;
    
        canvas.toBlob((blob) => {
            const reader = new FileReader();
            reader.onload = function() {
                // ✅ Envía SOLO el ArrayBuffer (no un objeto con metadatos)
                socket.emit('video_frame', reader.result); // reader.result es un ArrayBuffer
            };
            reader.onerror = () => console.error('Error al leer el blob');
            reader.readAsArrayBuffer(blob);
        }, 'image/jpeg', 0.8);
    }


    // Function to send frame to server
    function test() {
        socket.emit('test');
    }
    
    // Asegurarse de que las voces estén cargadas antes de usarlas
    if (speechSynthesis.onvoiceschanged !== undefined) {
        speechSynthesis.onvoiceschanged = function() {
            console.log('Voces cargadas');
        };
    }
    
    // Export the sendFrame function to be used by camera.js
    window.WebSocketClient = {
        sendFrame: sendFrame,
        test: test,
        speakText: speakText // Hacer la función accesible globalmente si es necesario
    };
});
