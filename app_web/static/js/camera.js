document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const startButton = document.getElementById('startButton');
    const statusDiv = document.getElementById('status');
    const resultDiv = document.getElementById('result');
    const confidenceDiv = document.getElementById('confidence');
    
    let stream = null;
    let isRunning = false;
    let animationId = null;
    let lastUpdate = 0;
    const PREDICTION_INTERVAL = 200; // ms

    // Iniciar cámara
    async function startCamera() {
        statusDiv.textContent = 'Solicitando acceso a la cámara...';
        
        try {
            // Primero verificar si el navegador soporta la API
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('Tu navegador no soporta el acceso a la cámara o está desactualizado.');
            }
            
            // Verificar si ya tenemos un stream
            if (stream) {
                stopCameraStream();
            }
            
            // Solicitar acceso a la cámara
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user',
                    frameRate: { ideal: 30 }
                },
                audio: false
            });
            
            // Configurar el elemento de video
            video.srcObject = stream;
            
            // Esperar a que el video esté listo
            return new Promise((resolve) => {
                video.onloadedmetadata = () => {
                    video.play();
                    startButton.disabled = false;
                    statusDiv.textContent = 'Cámara lista. Presiona Iniciar.';
                    resolve();
                };
            });
        } catch (err) {
            console.error("Error al acceder a la cámara:", err);
            let errorMessage = 'Error al acceder a la cámara: ';
            
            if (err.name === 'NotAllowedError') {
                errorMessage += 'Permiso denegado. Por favor, permite el acceso a la cámara.';
            } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
                errorMessage += 'No se encontró ninguna cámara conectada.';
            } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
                errorMessage += 'La cámara ya está en uso o no se puede acceder a ella.';
            } else {
                errorMessage += err.message;
            }
            
            statusDiv.textContent = errorMessage;
            startButton.disabled = true;
            throw err;
        }
    }
    
    // Detener el stream de la cámara
    function stopCameraStream() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
        if (video.srcObject) {
            video.srcObject = null;
        }
    }

    // Enviar frame al servidor para predicción
    async function processFrame() {
        if (!isRunning) return;
        
        const now = Date.now();
        
        // Solo procesar cada PREDICTION_INTERVAL ms
        if (now - lastUpdate >= PREDICTION_INTERVAL) {
            lastUpdate = now;
            
            // Limpiar el canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Dibujar el frame en el canvas (espejado)
            ctx.save();
            ctx.scale(-1, 1);  // Invertir horizontalmente
            ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
            ctx.restore();
            
            try {
                // Convertir a formato para enviar
                const imageData = canvas.toDataURL('image/jpeg', 0.8);
                const blob = await (await fetch(imageData)).blob();
                
                const formData = new FormData();
                formData.append('image', blob, 'frame.jpg');
                
                // Enviar al servidor
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    if (result.prediction!="") {
                        resultDiv.textContent = `Seña: ${result.prediction}`;
                    } else {
                        resultDiv.textContent = 'No se detectaron manos';
                    }
                }
                
            } catch (error) {
                console.error('Error:', error);
                statusDiv.textContent = `Error: ${error.message}`;
            }
        }
        
        animationId = requestAnimationFrame(processFrame);
    }

    // Control del botón de inicio/detención
    startButton.addEventListener('click', () => {
        if (isRunning) {
            isRunning = false;
            cancelAnimationFrame(animationId);
            startButton.textContent = 'Iniciar';
            statusDiv.textContent = 'Detenido';
            resultDiv.textContent = 'Esperando detección...';
            confidenceDiv.textContent = '';
        } else {
            isRunning = true;
            startButton.textContent = 'Detener';
            statusDiv.textContent = 'Detectando...';
            processFrame();
        }
    });



    // Iniciar cámara al cargar
    startCamera().catch(error => {
        console.error('Error al iniciar la cámara:', error);
    });
    
    // Manejar el cierre de la página para limpiar recursos
    window.addEventListener('beforeunload', () => {
        stopCameraStream();
        if (animationId) {
            cancelAnimationFrame(animationId);
        }
    });
});