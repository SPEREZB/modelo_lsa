// WebSocket client for LSP application
document.addEventListener('DOMContentLoaded', () => {
    // Connect to WebSocket server
    const socket = io();
    const statusDiv = document.getElementById('status');
    const resultDiv = document.getElementById('result');
    const confidenceDiv = document.getElementById('confidence');
    
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
    socket.on('prediction', (data) => {
        if (data.error) {
            console.error('Error del servidor:', data.error);
            statusDiv.textContent = `Error: ${data.error}`;
            return;
        }
        
        if (data.success) {
            const currentPrediction = data.prediction;
            resultDiv.textContent = `Letra: ${currentPrediction}`;
            confidenceDiv.textContent = `Confianza: ${(data.confidence * 100).toFixed(2)}%`;
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
    
    // Export the sendFrame function to be used by camera.js
    window.WebSocketClient = {
        sendFrame: sendFrame,
        test: test
    };
}); 