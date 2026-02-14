// static/js/sign_avatar.js
class SignAvatar {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
    }

    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.mediaRecorder = new MediaRecorder(stream);
            this.audioChunks = [];
            this.isRecording = true;
            
            this.mediaRecorder.ondataavailable = (event) => {
                this.audioChunks.push(event.data);
            };
            
            this.mediaRecorder.onstop = this.processRecording.bind(this);
            this.mediaRecorder.start();
            
        } catch (error) {
            console.error('Error al acceder al micrófono:', error);
        }
    }

    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            
            // Detener todas las pistas del stream
            this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
    }

    async processRecording() {
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.wav');

        try {
            const response = await fetch('/sign_avatar/process_audio', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            
            if (result.status === 'success') {
                this.playAnimation(result.animation_frames);
            } else {
                console.error('Error al procesar el audio:', result.error);
            }
        } catch (error) {
            console.error('Error al enviar el audio:', error);
        }
    }

    playAnimation(frames) {
        // Aquí iría la lógica para mostrar la animación
        console.log('Reproduciendo animación con', frames.length, 'frames');
        // Implementar la visualización de la animación
    }
}

// Uso:
// const avatar = new SignAvatar('avatar-container');
// document.getElementById('start-btn').addEventListener('click', () => avatar.startRecording());
// document.getElementById('stop-btn').addEventListener('click', () => avatar.stopRecording());