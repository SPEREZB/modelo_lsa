document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const startButton = document.getElementById('startButton');
    const switchCameraBtn = document.getElementById('switchCamera');
    const statusDiv = document.getElementById('status');
    const resultDiv = document.getElementById('result');
    const confidenceDiv = document.getElementById('confidence');
    
    let stream = null;
    let isRunning = false;
    let animationId = null;
    let lastUpdate = 0;
    const PREDICTION_INTERVAL = 100; // ms - reducido para mejor respuesta
    let webSocketClient = null;
    
    // Variables para manejo de cámaras
    let currentFacingMode = 'user'; // 'user' para frontal, 'environment' para trasera
    let devices = [];
    let currentDeviceId = '';

    // Obtener lista de dispositivos de cámara
    async function getCameraDevices() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
            console.log('enumerateDevices() no es soportado en este navegador');
            return [];
        }
        
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            return devices.filter(device => device.kind === 'videoinput');
        } catch (err) {
            console.error('Error al enumerar dispositivos:', err);
            return [];
        }
    }
    
    // Iniciar cámara
    async function startCamera(facingMode = 'user') {
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
            
            // Obtener dispositivos de cámara
            devices = await getCameraDevices();
            
            // Si hay múltiples cámaras, mostrar el botón de cambio
            if (devices.length > 1 && switchCameraBtn) {
                switchCameraBtn.style.display = 'inline-block';
            } else if (switchCameraBtn) {
                switchCameraBtn.style.display = 'none';
            }
            
            // Configuración de la cámara
            const constraints = {
                video: { 
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    frameRate: { ideal: 30 },
                    facingMode: { exact: facingMode }
                },
                audio: false
            };
            
            // Si ya tenemos un deviceId específico, usarlo
            if (currentDeviceId) {
                delete constraints.video.facingMode;
                constraints.video.deviceId = { exact: currentDeviceId };
            }
            
            // Solicitar acceso a la cámara
            stream = await navigator.mediaDevices.getUserMedia(constraints);
            
            // Actualizar el modo de cámara actual
            currentFacingMode = facingMode;
            
            // Configurar el elemento de video
            video.srcObject = stream;

            const shouldMirror = (currentDeviceId && currentFacingMode === 'user') || (!currentDeviceId && facingMode === 'user');
            video.style.transform = shouldMirror ? 'scaleX(-1)' : 'scaleX(1)';
            
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

    // Función para procesar y enviar frames al servidor
    async function processFrame() { 
        if (!isRunning) return;
        
        const now = Date.now();
        
        // Solo procesar cada PREDICTION_INTERVAL ms
        if (now - lastUpdate >= PREDICTION_INTERVAL) {
            lastUpdate = now;
            
            // Asegurarse de que el canvas tenga el mismo tamaño que el video
            if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
            }

            // Limpiar el canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Determinar si estamos usando la cámara frontal o trasera
            const isFrontCamera = currentFacingMode === 'user' || 
                                (currentDeviceId && devices.find(d => d.deviceId === currentDeviceId)?.label.toLowerCase().includes('front'));

            // Aplicar transformaciones según la cámara
            ctx.save();
            
            if (isFrontCamera) {
                // Para cámara frontal: aplicar espejo horizontal
                ctx.scale(-1, 1);
                ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
            } else {
                // Para cámara trasera: sin espejo
                ctx.scale(-1, 1);
                ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);

               // ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            }
            
            ctx.restore();

            // Enviar el frame a través del WebSocket si está disponible
            try {
                if (window.WebSocketClient && window.WebSocketClient.sendFrame) {
                    window.WebSocketClient.sendFrame(canvas);
                }
            } catch (error) {
                console.error('Error enviando frame:', error);
            }
        }
        
        animationId = requestAnimationFrame(processFrame);
    }
    // Función para cambiar entre cámaras
    async function switchCamera() {
        if (devices.length < 2) return;
        
        // Guardar el estado actual de ejecución
        const wasRunning = isRunning;
        
        // Detener la cámara actual
        if (wasRunning) {
            stopPrediction();
        }
        
        try {
            // Encontrar el índice de la cámara actual
            const currentIndex = devices.findIndex(device => 
                device.deviceId === currentDeviceId
            );
            
            // Calcular el índice de la siguiente cámara
            const nextIndex = (currentIndex + 1) % devices.length;
            const nextDevice = devices[nextIndex];
            
            // Determinar si la cámara es frontal o trasera basado en la etiqueta del dispositivo
            const deviceLabel = nextDevice.label.toLowerCase();
            const isFrontFacing = deviceLabel.includes('front') || 
                                deviceLabel.includes('user') ||
                                deviceLabel.includes('facing');
            
            // Actualizar el modo de la cámara
            currentFacingMode = isFrontFacing ? 'user' : 'environment';
            currentDeviceId = nextDevice.deviceId;
            
            // Detener la cámara actual
            stopCameraStream();
            
            // Iniciar la nueva cámara
            await startCamera(currentFacingMode);
            
            // Reanudar la predicción si estaba activa
            if (wasRunning) {
                startPrediction();
            }
            
            console.log('Cambiando a cámara:', {
                deviceId: currentDeviceId,
                label: nextDevice.label,
                facingMode: currentFacingMode,
                isFrontFacing: isFrontFacing
            });
            
        } catch (error) {
            console.error('Error al cambiar de cámara:', error);
            // Restaurar el estado en caso de error
            if (wasRunning) {
                startPrediction();
            }
        }
    }
    // Control del botón de cambio de cámara
    if (switchCameraBtn) {
        switchCameraBtn.addEventListener('click', switchCamera);
    }
    
    // Control del botón de inicio/detención
    startButton.addEventListener('click', () => { 
        if (isRunning) { 
            isRunning = false;
            cancelAnimationFrame(animationId);
            startButton.textContent = 'Iniciar Detección';
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