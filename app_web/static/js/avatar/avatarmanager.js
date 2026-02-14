class AvatarManager {
    constructor(scene) {
        this.scene = scene;
        this.avatar = null;
        this.mixer = null;
        this.animations = {};
    }

    async loadModel() {
        try {
            const loader = new THREE.GLTFLoader();
            
            // Cargar modelo base
            const gltf = await loader.loadAsync('/static/models/avatar/final.glb');
            this.avatar = gltf.scene;
            this.scene.add(this.avatar);
            
            // Configurar mixer
            this.mixer = new THREE.AnimationMixer(this.avatar);
            
            // Cargar animaciones
            await this.loadAnimation('idle', '/static/models/avatar/animations/idle.glb');
            
            // Reproducir animación en reposo
            this.playAnimation('idle', true);
            
            return this.avatar;
        } catch (error) {
            console.error('Error loading model:', error);
            throw error;
        }
    }

async loadAnimation(name, path) {
    try {
        console.log(`[DEBUG] Cargando animación: ${name} desde ${path}`);
        const loader = new THREE.GLTFLoader();
        const gltf = await loader.loadAsync(path);
        
        console.log(`[DEBUG] Modelo cargado para animación ${name}:`, gltf);
        
        if (gltf.animations && gltf.animations.length > 0) {
            console.log(`[DEBUG] Animaciones encontradas:`, gltf.animations.map(a => a.name));
            this.animations[name] = gltf.animations[0];
            console.log(`[DEBUG] Animación "${name}" asignada correctamente`);
        } else {
            console.warn(`[WARN] No se encontraron animaciones en ${path}`);
            // Crear una animación vacía para evitar errores
            this.animations[name] = new THREE.AnimationClip(name, 1, []);
        }
    } catch (error) {
        console.error(`[ERROR] Error cargando animación ${name}:`, error);
        // Crear una animación vacía para evitar errores
        this.animations[name] = new THREE.AnimationClip(name, 1, []);
    }
}

    playAnimation(name, loop = true) {
    console.log(`[DEBUG] Intentando reproducir animación: ${name}`);
    console.log(`[DEBUG] Animaciones cargadas:`, Object.keys(this.animations));
    console.log(`[DEBUG] Mixer disponible:`, this.mixer ? 'Sí' : 'No');
    
    if (!this.animations[name]) {
        console.warn(`[WARN] La animación "${name}" no existe en this.animations`);
        return null;
    }
    
    if (!this.mixer) {
        console.warn('[WARN] El mixer no está inicializado');
        return null;
    }

    // Detener animación actual
    if (this.currentAction) {
        console.log('[DEBUG] Deteniendo animación actual');
        this.currentAction.stop();
    }

    console.log(`[DEBUG] Iniciando animación: ${name}`);
    const clip = this.animations[name];
    const action = this.mixer.clipAction(clip);
    
    if (loop) {
        action.setLoop(THREE.LoopRepeat);
    } else {
        action.setLoop(THREE.LoopOnce);
        action.clampWhenFinished = true;
    }
    
    action.reset().play();
    this.currentAction = action;
    console.log(`[DEBUG] Animación "${name}" iniciada correctamente`);
    return action;
}
    update(delta) {
        if (this.mixer) {
            this.mixer.update(delta);
        }
    }
}