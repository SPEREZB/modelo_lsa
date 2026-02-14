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
    const MIN_TIME_BETWEEN_SPEECH_MS = 3000; // 3 segundos entre mensajes

    // Asegurarse de que las voces estén cargadas antes de usarlas
    // ✅ AHORA ESTÁ DENTRO DEL DOMContentLoaded
    if ('speechSynthesis' in window && window.speechSynthesis.onvoiceschanged !== undefined) {
        window.speechSynthesis.onvoiceschanged = function () {
            console.log('Voces cargadas:', window.speechSynthesis.getVoices().length);
        };
    }

    // Connection status
    socket.on('connect', () => {
        console.log('Conectado al servidor WebSocket');
        statusDiv.textContent = 'Conectado al servidor de reconocimiento';

        // Export the sendFrame function to be used by camera.js
        window.WebSocketClient = {
            sendFrame: sendFrame,
            test: test,
            speakText: speakText
        };
    });

    socket.on('disconnect', () => {
        console.log('Desconectado del servidor WebSocket');
        statusDiv.textContent = 'Desconectado del servidor';
    });

    socket.on('connect_error', (error) => {
        console.error('Error de conexión WebSocket:', error);
        statusDiv.textContent = 'Error de conexión con el servidor';
    });

    // Función para leer texto en voz alta
    // function speakText(text) {
    //     if (window.flutter_inappwebview) {
    //         window.flutter_inappwebview.callHandler('speakText', text);
    //     } else {
    //         console.warn('TTS channel no disponible');
    //     }
    // }
    // Función para reproducir texto usando Flutter TTS si está disponible
    function speakText(text) {
        if (window.flutter_inappwebview && window.flutter_inappwebview.callHandler) {
            // 🔥 Llamamos al handler de Flutter
            window.flutter_inappwebview.callHandler('speakText', text);
        } else if ('speechSynthesis' in window) {
            // fallback para navegadores normales
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = 'es-ES';
            utterance.rate = 0.9;
            utterance.pitch = 1.0;

            const voices = window.speechSynthesis.getVoices();
            const spanishVoice = voices.find(voice => voice.lang.startsWith('es'));
            if (spanishVoice) utterance.voice = spanishVoice;

            window.speechSynthesis.speak(utterance);
        } else {
            console.warn('TTS no disponible ni en Flutter ni en navegador');
        }
    }



    // Handle prediction responses
    socket.on('prediction', (data) => {
        if (data.error) {
            console.error('Error del servidor:', data.error);
            statusDiv.textContent = `Error: ${data.error}`;
            return;
        }

        if (data.success) {
            if (data.debug) {
                const st = data.debug.state ?? '';
                const frames = data.debug.frames ?? '';
                const total = data.debug.sequence_length ?? '';
                const motion = (data.debug.motion ?? 0).toFixed(6);
                const cd = (data.debug.cooldown_remaining ?? 0).toFixed(2);
                statusDiv.textContent = `Estado: ${st} | Frames: ${frames}/${total} | Motion: ${motion} | Cooldown: ${cd}s`;
            }

            const currentPrediction = data.prediction;  // Solo el texto de la predicción

            if (currentPrediction) {
                const predictionText = `Seña: ${currentPrediction}`;
                resultDiv.textContent = predictionText;
                confidenceDiv.textContent = `Confianza: ${(data.confidence * 100).toFixed(2)}%`;
            } else {
                console.log('Sin predicción confirmada aún. Debug:', data.debug);
                return;
            }

            // Verificar si la predicción es nueva y ha pasado el tiempo mínimo
            const currentTime = Date.now();
            const isNewPrediction = currentPrediction !== lastSpokenPrediction;
            const hasEnoughTimePassed = (currentTime - lastSpokenTime) > MIN_TIME_BETWEEN_SPEECH_MS;

            if (isNewPrediction && hasEnoughTimePassed) {
                console.log('Intentando hablar:', currentPrediction);
                speakText(currentPrediction);
                lastSpokenPrediction = currentPrediction;  // Solo guardamos la predicción, sin el prefijo
                lastSpokenTime = currentTime;
            }
        }
    });

    // Function to send frame to server
    // Optimizado: calidad reducida para menor latencia
    function sendFrame(canvas) {
        if (!socket.connected) return;

        canvas.toBlob((blob) => {
            if (!blob) return;
            const reader = new FileReader();
            reader.onload = function () {
                socket.emit('video_frame', reader.result);
            };
            reader.readAsArrayBuffer(blob);
        }, 'image/jpeg', 0.6); // Calidad reducida de 0.8 a 0.6 para menor latencia
    }

    // Function to send test signal
    function test() {
        socket.emit('test');
    }

});