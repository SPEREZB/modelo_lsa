class SignAvatar {
     // Función de suavizado para transiciones
    easeInOutQuad(t) {
        return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
    }

    // Estado para transiciones suaves entre keypoints
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.clock = new THREE.Clock();
        this.animationMixer = null;
        this.animations = {};
        this.currentAnimation = null;
        this.currentKeypoints = null;
        this.isPlayingSequence = false;
        this.animationFrameId = null;
        this.isApplyingKeypoints = false; // Flag para controlar si se están aplicando keypoints manuales
        
        // Estado para transiciones suaves de keypoints
        this.isTransitioning = false;
        this.currentHandPose = null; // Almacena la pose actual de los dedos
        this.transitionDuration = 400; // Duración de transición en ms
        
        // Mover boneMap al principio
        this.boneMap = {
            // Brazo derecho
            rightShoulder: 'mixamorigRightShoulder',
            rightUpperArm: 'mixamorigRightArm',
            rightForearm: 'mixamorigRightForeArm',
            rightHand: 'mixamorigRightHand',

            // Brazo izquierdo
            leftShoulder: 'mixamorigLeftShoulder',
            leftUpperArm: 'mixamorigLeftArm',
            leftForearm: 'mixamorigLeftForeArm',
            leftHand: 'mixamorigLeftHand'
        };

        this.currentGender = 'male';

        this.initThreeJS();
        this.loadModel(this.currentGender);
        this.setupUI();
    }

// ---------------------
// Pose relajada de brazos (brazos abajo junto al cuerpo)
// ---------------------
async setArmsRelaxedPose(duration = 300) {
    try {
        // Usamos los huesos mapeados en setupModelParts()
        const L = this.bones && this.bones.leftUpperArm;
        const LF = this.bones && this.bones.leftForearm;
        const LH = this.bones && this.bones.leftHand;
        const R = this.bones && this.bones.rightUpperArm;
        const RF = this.bones && this.bones.rightForearm;
        const RH = this.bones && this.bones.rightHand;

        const tasks = [];
        // Detectar si estamos usando rig CC_Base (nuevo avatar)
        const isCCBaseRig = !!(R && typeof R.name === 'string' && R.name.indexOf('CC_Base_') === 0);

        if (isCCBaseRig) {
            // Para CC_Base: brazos abajo junto al cuerpo.
            // Partimos de la T-pose (brazos en cruz) y rotamos hacia abajo.
            // Estos ángulos están ajustados pensando en CC_Base_R_Upperarm / CC_Base_L_Upperarm.
            if (L) tasks.push(this.animateBoneQuaternion(L, { x: 0, y: 0, z: -1.2 }, duration));
            if (R) tasks.push(this.animateBoneQuaternion(R, { x: 0, y: 0, z: 1.2 }, duration));
        } else {
            // Lógica original (Mixamo): detectar si el brazo está hacia adelante
            // y bajar brazos por rotación sobre Z.
            let forwardPose = false;
            try {
                const rShoulder = this.bones && this.bones.rightShoulder;
                const rHand = this.bones && this.bones.rightHand;
                if (rShoulder && rHand) {
                    const ps = new THREE.Vector3();
                    const ph = new THREE.Vector3();
                    rShoulder.getWorldPosition(ps);
                    rHand.getWorldPosition(ph);
                    const v = ph.sub(ps).normalize();
                    forwardPose = Math.abs(v.z) >= Math.abs(v.x) && Math.abs(v.z) >= Math.abs(v.y) && v.z > 0;
                }
            } catch (_) {}

            if (forwardPose) {
                if (L) tasks.push(this.animateBoneQuaternion(L, { x: 1.35, y: 0, z: 0 }, duration));
                if (R) tasks.push(this.animateBoneQuaternion(R, { x: 1.35, y: 0, z: 0 }, duration));
            } else {
                if (L) tasks.push(this.animateBoneQuaternion(L, { x: 0, y: 0, z: 1.57 }, duration));
                if (R) tasks.push(this.animateBoneQuaternion(R, { x: 0, y: 0, z: -1.57 }, duration));
            }
        }
        if (LF) tasks.push(this.animateBoneQuaternion(LF, { x: 0, y: 0, z: 0 }, duration));
        if (LH) tasks.push(this.animateBoneQuaternion(LH, { x: 0, y: 0, z: 0 }, duration));
        if (RF) tasks.push(this.animateBoneQuaternion(RF, { x: 0, y: 0, z: 0 }, duration));
        if (RH) tasks.push(this.animateBoneQuaternion(RH, { x: 0, y: 0, z: 0 }, duration));

        await Promise.all(tasks);
        // Al volver a la pose relajada, limpiamos el estado de orientación de la palma
        this._palmOrientationMode = null;
    } catch (e) {
        console.warn('[WARN] No se pudo aplicar la pose relajada:', e);
    }
}


    async loadModel(gender = 'male') {
        const fileName = gender === 'female' ? 'bodymujer.glb' : 'AvatarManosGrandes.glb';
        const modelPath = `/static/models/avatar/${fileName}`;
        console.log(`[DEBUG] Loading model from: ${modelPath}`);
        
        try {
            const loader = new THREE.GLTFLoader();
            const gltf = await loader.loadAsync(modelPath);
            
            this.avatar = gltf.scene;
            this.avatar.scale.set(0.1, 0.1, 0.1);
            this.avatar.position.set(0, -5, 0);
            
            // Set up the animation mixer
            this.animationMixer = new THREE.AnimationMixer(this.avatar);
            
            // Store animations if any
            if (gltf.animations && gltf.animations.length) {
                gltf.animations.forEach(anim => {
                    this.animations[anim.name] = anim;
                });
            }
            
            // Set up the skeleton helper for debugging
            const skeletonHelper = new THREE.SkeletonHelper(this.avatar);
            skeletonHelper.visible = false; // Set to false to hide skeleton
            this.scene.add(skeletonHelper);
            
            // Set up the model parts
            this.setupModelParts();
            
            this.scene.add(this.avatar);
            
            // Pose inicial: brazos relajados hacia abajo
            await this.setArmsRelaxedPose(0);

            // Set up camera to view the model
            this.setupCameraForModel(); 

            this.bones = [];
            this.avatar.traverse((object) => {
                if (object.isBone) this.bones.push(object);
            });
          
        } catch (error) {
            console.error('[ERROR] Error loading model:', error);
            this.addTestCube();
        }
    }

    async setGender(gender) {
        const normalized = gender === 'female' ? 'female' : 'male';
        if (normalized === this.currentGender && this.avatar) {
            return;
        }
        this.currentGender = normalized;
        if (this.avatar) {
            if (this.animationMixer) {
                this.animationMixer.stopAllAction();
                this.animationMixer.uncacheRoot(this.avatar);
                this.animationMixer = null;
            }
            this.scene.remove(this.avatar);
            this.avatar.traverse((object) => {
                if (object.isMesh) {
                    if (object.geometry) {
                        object.geometry.dispose();
                    }
                    if (Array.isArray(object.material)) {
                        object.material.forEach(m => m && m.dispose && m.dispose());
                    } else if (object.material && object.material.dispose) {
                        object.material.dispose();
                    }
                }
            });
            this.avatar = null;
        }
        await this.loadModel(this.currentGender);
    }

    setupModelParts() {
        // Helper function to find bones by name pattern
        // --- Depuración: listar todos los huesos ---
console.log('=== LISTA COMPLETA DE HUESOS ===');
this.avatar.traverse((node) => {
    if (node.isBone) console.log(node.name);
});



        const findBone = (root, names) => {
            let exactMatch = null;
            let partialMatch = null;

            root.traverse((node) => {
                if (!node.isBone || !node.name) return;

                const lowerName = node.name.toLowerCase();
                for (const rawName of names) {
                    const target = rawName.toLowerCase();

                    // Coincidencia exacta de nombre de hueso (preferida)
                    if (!exactMatch && lowerName === target) {
                        exactMatch = node;
                        return;
                    }

                    // Coincidencia parcial solo como respaldo
                    if (!partialMatch && lowerName.includes(target)) {
                        partialMatch = node;
                    }
                }
            });

            return exactMatch || partialMatch;
        };

        // Find all the necessary bones
        this.bones = {
            // Left Arm
            leftShoulder: findBone(this.avatar, ['leftshoulder', 'shoulder_l', 'cc_base_l_clavicle', 'l_clavicle']),
            leftUpperArm: findBone(this.avatar, ['leftarm', 'upperarm_l', 'cc_base_l_upperarm', 'l_upperarm', 'upperarm']),
            leftForearm: findBone(this.avatar, ['leftforearm', 'forearm_l', 'cc_base_l_forearm', 'l_forearm', 'forearm']),
            leftHand: findBone(this.avatar, ['lefthand', 'hand_l', 'cc_base_l_hand', 'l_hand', 'hand']),
            
            // Right Arm
            rightShoulder: findBone(this.avatar, ['rightshoulder', 'shoulder_r', 'cc_base_r_clavicle', 'r_clavicle']),
            rightUpperArm: findBone(this.avatar, ['rightarm', 'upperarm_r', 'cc_base_r_upperarm', 'r_upperarm', 'upperarm']),
            rightForearm: findBone(this.avatar, ['rightforearm', 'forearm_r', 'cc_base_r_forearm', 'r_forearm', 'forearm']),
            rightHand: findBone(this.avatar, ['righthand', 'hand_r', 'cc_base_r_hand', 'r_hand', 'hand']),
            
            // Head and neck
            head: findBone(this.avatar, ['head']),
            neck: findBone(this.avatar, ['neck'])
        };
 
    }

    // Move a specific bone
    async moveBone(boneName, position = null, rotation = null, duration = 500) {
    if (!this.bones || this.bones.length === 0) {
        console.error('[ERROR] No se encontraron huesos cargados');
        return;
    }

    // Reemplazar por el nombre real de Mixamo si existe en el mapeo
    const realName = this.boneMap[boneName] || boneName;
    const bone = this.bones.find(b => b.name === realName);

    if (!bone) {
        console.warn(`Bone not found: ${realName}`);
        return;
    }

    // Aplicar rotación
    if (rotation) {
        const startRot = bone.rotation.clone();
        const endRot = new THREE.Euler(
            startRot.x + (rotation.x || 0),
            startRot.y + (rotation.y || 0),
            startRot.z + (rotation.z || 0)
        );

        const startTime = performance.now();
        const animate = (time) => {
            const t = Math.min((time - startTime) / duration, 1);
            bone.rotation.set(
                startRot.x + (endRot.x - startRot.x) * t,
                startRot.y + (endRot.y - startRot.y) * t,
                startRot.z + (endRot.z - startRot.z) * t
            );
            if (t < 1) requestAnimationFrame(animate);
        };
        requestAnimationFrame(animate);
    }

    // Aplicar posición opcional
    if (position) {
        const startPos = bone.position.clone();
        const endPos = startPos.clone().add(new THREE.Vector3(
            position.x || 0, position.y || 0, position.z || 0
        ));

        const startTime = performance.now();
        const animatePos = (time) => {
            const t = Math.min((time - startTime) / duration, 1);
            bone.position.lerpVectors(startPos, endPos, t);
            if (t < 1) requestAnimationFrame(animatePos);
        };
        requestAnimationFrame(animatePos);
    }
}



    // Setup Three.js scene
    initThreeJS() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xf0f0f0);
        
        // Medir tamaño real del contenedor para calcular aspecto
        const width = this.container.offsetWidth || this.container.clientWidth || window.innerWidth;
        const height = this.container.offsetHeight || this.container.clientHeight || window.innerHeight || 1;
        const aspect = width / height;

        // Camera
        this.camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 1000);
        this.camera.position.set(0, 5, 3);
        
        // Renderer (mejor nitidez en pantallas de alta densidad, p.ej. celulares)
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        this.renderer.setSize(width, height);
        this.renderer.shadowMap.enabled = true;
        this.container.appendChild(this.renderer.domElement);
        
        // Lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(1, 1, 1);
        this.scene.add(directionalLight);
        
        // Grid helper
        const gridHelper = new THREE.GridHelper(10, 10);
        gridHelper.visible = false;
        this.scene.add(gridHelper);
        
        // Orbit controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableRotate = false;
        this.controls.enableZoom = false;
        this.controls.enablePan = false;
        this.controls.target.set(0, 1, 0);
        this.controls.update();
        
        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());
        
        // Start animation loop
        this.animate();
    }

   setupCameraForModel() {
    if (!this.avatar || !this.camera) return;

    const isMobile = window.innerWidth <= 768;

    // Calcular el tamaño y centro del modelo
    const box = new THREE.Box3().setFromObject(this.avatar);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());

    // Elevar el encuadre hacia torso/cabeza
    // IMPORTANTE CAMARA
    // distance controla qué tan lejos está la cámara del avatar.
    //  - Valores MÁS PEQUEÑOS  => cámara más cerca  => avatar más grande (más zoom).
    //  - Valores MÁS GRANDES   => cámara más lejos => avatar más pequeño (menos zoom).
    const distance = Math.max(size.y * 0.1, 0.13);

    // baseHeightOffset desplaza hacia ARRIBA o ABAJO TODO el conjunto cámara+target
    // sin cambiar el ángulo: se suma igual a cámara y a target.
    //  - Valores MÁS GRANDES  => todo sube (como pasar de 2m a 3m de altura).
    //  - Valores MÁS PEQUEÑOS => todo baja.
    const baseHeightOffset = size.y * 1;

    // Altura base del centro del modelo (ligeramente sobre el centro real)
    const baseTargetY = center.y + 0.03 + baseHeightOffset;
    const target = new THREE.Vector3(center.x, baseTargetY, center.z);

    // heightFactor controla SOLO la diferencia de altura entre cámara y target (ángulo vertical).
    //  - Si heightFactor = 0  => cámara y target a la misma altura (la cámara mira “derecho”).
    //  - Si heightFactor > 0  => cámara más alta que el target (mira un poco hacia abajo).
    //  - Si heightFactor < 0  => cámara más baja que el target (mira un poco hacia arriba).
    const heightFactor = 0; // puedes ajustar este valor si luego quieres cambiar el ángulo

    this.camera.position.set(
        target.x,
        target.y + size.y * heightFactor,
        target.z + distance * 0.7
    );

    this.camera.lookAt(target);

    if (this.controls) {
        this.controls.target.copy(target);
        this.controls.update();
    }
    // Campo de visión algo más estrecho para aumentar el efecto de zoom
    this.camera.fov = 24;
    this.camera.updateProjectionMatrix();
    this.setArmsRelaxedPose(0);
}


    onWindowResize() {
        const width = this.container.offsetWidth || this.container.clientWidth || window.innerWidth;
        const height = this.container.offsetHeight || this.container.clientHeight || window.innerHeight || 1;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();

        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        this.renderer.setSize(width, height);
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        
        const delta = this.clock.getDelta();
        
        // Solo actualizar el animationMixer si NO se están aplicando keypoints manuales
        if (this.animationMixer && !this.isApplyingKeypoints) {
            this.animationMixer.update(delta);
        }
        
        if (this.controls) {
            this.controls.update();
        }
        
        this.renderer.render(this.scene, this.camera);
    }

 
  // posicionar mano en frente del avatar
findBoneByContains(nameFragment) {
    if (!this.bones || this.bones.length === 0) return null;
    return this.bones.find(b => b.name.toLowerCase().includes(nameFragment.toLowerCase())) || null;
}

// ---------------------
// helper: animar rotación usando quaternions (slerp)
// ---------------------
animateBoneQuaternion(bone, targetEuler, duration = 600) {
    return new Promise(resolve => {
        if (!bone) return resolve();

        // Guardar quaternion inicial y objetivo
        const startQ = bone.quaternion.clone();
        const targetQ = new THREE.Quaternion().setFromEuler(new THREE.Euler(
            targetEuler.x || 0,
            targetEuler.y || 0,
            targetEuler.z || 0,
            'XYZ'
        ));

        const startTime = performance.now();
        const step = (now) => {
            const t = Math.min((now - startTime) / duration, 1);
            THREE.Quaternion.slerp(startQ, targetQ, bone.quaternion, t);
            bone.updateMatrixWorld(true);
            if (t < 1) requestAnimationFrame(step);
            else resolve();
        };
        requestAnimationFrame(step);
    });
}

// IMPORTANTE: Extiende el brazo derecho frente al pecho para hacer señas (posición base de señalización)
async poseRightArmForAMi(duration = 800) {
 
    const shoulder = this.findBoneByContains('rightshoulder')
        || this.findBoneByContains('mixamorigRightShoulder')
        || this.findBoneByContains('cc_base_r_clavicle')
        || this.findBoneByContains('r_clavicle');
    const upperArm = this.findBoneByContains('rightarm')
        || this.findBoneByContains('mixamorigRightArm')
        || this.findBoneByContains('cc_base_r_upperarm')
        || this.findBoneByContains('r_upperarm');
    const forearm = this.findBoneByContains('rightforearm')
        || this.findBoneByContains('mixamorigRightForeArm')
        || this.findBoneByContains('cc_base_r_forearm')
        || this.findBoneByContains('r_forearm');
    const hand = this.findBoneByContains('righthand')
        || this.findBoneByContains('mixamorigRightHand')
        || this.findBoneByContains('cc_base_r_hand')
        || this.findBoneByContains('r_hand');

    if (!upperArm || !forearm || !hand) {
        console.warn('[WARN] No se detectaron todos los huesos del brazo derecho.');
        return;
    }

    if (!this._prevRightArmPose) this._prevRightArmPose = {};
    this._prevRightArmPose = {
        shoulder: shoulder ? { name: shoulder.name, q: shoulder.quaternion.clone() } : null,
        upperArm: upperArm ? { name: upperArm.name, q: upperArm.quaternion.clone() } : null,
        forearm: forearm ? { name: forearm.name, q: forearm.quaternion.clone() } : null,
        hand: hand ? { name: hand.name, q: hand.quaternion.clone() } : null
    };

    // Ajustar objetivos según el tipo de rig
    const isCCBaseRig = !!(upperArm && typeof upperArm.name === 'string' && upperArm.name.indexOf('CC_Base_') === 0);

    let shoulderTarget, upperArmTarget, forearmTarget, handTarget;

    if (isCCBaseRig) {
        // Para CC_Base: mantener el brazo a media altura, ligeramente delante del pecho.
        // Estos valores pueden necesitar pequeños ajustes visuales, pero evitan levantar el brazo por encima de la cabeza.
       const _shoulderTarget = { x: 1.7, y: 0, z: 1 };     // hombro ligeramente hacia adelante y centro
        const _upperArmTarget = { x: 0.3, y: 0.6, z: -1.1 };     // brazo extendido hacia el pecho
        const _forearmTarget = { x: 0, y: 0.6, z: 0.2 };      // doblar antebrazo hacia el pecho
        const _handTarget = { x: 1.4, y: 0, z: -1.9 };         // rotar muñeca 90° para orientar palma al frente

        shoulderTarget = _shoulderTarget;
        upperArmTarget = _upperArmTarget;
        forearmTarget = _forearmTarget;
        handTarget = _handTarget; 
    } else {
        // Valores originales pensados para Mixamo
        const _shoulderTarget = { x: 1.7, y: 0, z: 1 };     // hombro ligeramente hacia adelante y centro
        const _upperArmTarget = { x: 0.3, y: 0.6, z: -1.1 };     // brazo extendido hacia el pecho
        const _forearmTarget = { x: 0, y: 0.6, z: 0.2 };      // doblar antebrazo hacia el pecho
        const _handTarget = { x: 1.4, y: 0, z: -1.9 };         // rotar muñeca 90° para orientar palma al frente

        shoulderTarget = _shoulderTarget;
        upperArmTarget = _upperArmTarget;
        forearmTarget = _forearmTarget;
        handTarget = _handTarget; 
    }

    await Promise.all([
        shoulder ? this.animateBoneQuaternion(shoulder, shoulderTarget, duration) : Promise.resolve(),
        upperArm ? this.animateBoneQuaternion(upperArm, upperArmTarget, duration) : Promise.resolve(),
        forearm ? this.animateBoneQuaternion(forearm, forearmTarget, duration) : Promise.resolve(),
        hand ? this.animateBoneQuaternion(hand, handTarget, duration) : Promise.resolve()
    ]); 
}

// IMPORTANTE: Captura la pose inicial del brazo (brazos abajo) antes de una secuencia de señas
captureInitialArmPose() {
    if (this._initialRightArmPose) return; // Ya está capturada
    const shoulder = this.findBoneByContains('rightshoulder')
        || this.findBoneByContains('mixamorigRightShoulder')
        || this.findBoneByContains('cc_base_r_clavicle')
        || this.findBoneByContains('r_clavicle');
    const upperArm = this.findBoneByContains('rightarm')
        || this.findBoneByContains('mixamorigRightArm')
        || this.findBoneByContains('cc_base_r_upperarm')
        || this.findBoneByContains('r_upperarm');
    const forearm = this.findBoneByContains('rightforearm')
        || this.findBoneByContains('mixamorigRightForeArm')
        || this.findBoneByContains('cc_base_r_forearm')
        || this.findBoneByContains('r_forearm');
    const hand = this.findBoneByContains('righthand')
        || this.findBoneByContains('mixamorigRightHand')
        || this.findBoneByContains('cc_base_r_hand')
        || this.findBoneByContains('r_hand');
    this._initialRightArmPose = {
        shoulder: shoulder ? { name: shoulder.name, q: shoulder.quaternion.clone() } : null,
        upperArm: upperArm ? { name: upperArm.name, q: upperArm.quaternion.clone() } : null,
        forearm: forearm ? { name: forearm.name, q: forearm.quaternion.clone() } : null,
        hand: hand ? { name: hand.name, q: hand.quaternion.clone() } : null
    };
}

// IMPORTANTE: Limpia la pose inicial capturada (llamar después de restaurar el brazo)
clearInitialArmPose() {
    this._initialRightArmPose = null;
}

// IMPORTANTE: Resetea el modo de orientación de palma (llamar al final de una secuencia)
resetPalmOrientationMode() {
    this._palmOrientationMode = null;
}

// IMPORTANTE: Restaura el brazo derecho a su posición original (brazos abajo) después de hacer señas
async restoreRightArmPose(duration = 600, toInitial = false) {
    // Si toInitial es true, restaurar a la pose inicial capturada; si no, a la pose previa
    const data = toInitial ? this._initialRightArmPose : this._prevRightArmPose;
    if (!data) return;
    const fb = (n) => this.findBoneByContains(n) || null;
    const findByName = (name) => {
        const b = fb(name.toLowerCase());
        if (b && b.name === name) return b;
        return this.bones && Array.isArray(this.bones) ? this.bones.find(x => x.name === name) || null : null;
    };
    const tasks = [];
    const apply = (slot) => {
        if (!slot) return;
        const bone = findByName(slot.name);
        if (!bone) return;
        const e = new THREE.Euler().setFromQuaternion(slot.q, 'XYZ');
        tasks.push(this.animateBoneQuaternion(bone, { x: e.x, y: e.y, z: e.z }, duration));
    };
    apply(data.shoulder);
    apply(data.upperArm);
    apply(data.forearm);
    apply(data.hand);
    await Promise.all(tasks);
}



 // Animar la letra "A" en lengua de señas (cerrar puño)
async signLetterA() { 
    await this.poseRightArmForAMi(600);

    const findBone = (name) =>
        this.bones.find(b => b.name.includes(name)) || null;

    // Huesos de la mano derecha (según tu modelo Mixamo)
    const fingers = {
        thumb1: findBone('mixamorigRightHandThumb1') || findBone('CC_Base_R_Thumb1'),
        thumb2: findBone('mixamorigRightHandThumb2') || findBone('CC_Base_R_Thumb2'),
        thumb3: findBone('mixamorigRightHandThumb3') || findBone('CC_Base_R_Thumb3'),
        thumb4: findBone('mixamorigRightHandThumb4') || findBone('CC_Base_R_Thumb3'),

        index1: findBone('mixamorigRightHandIndex1') || findBone('CC_Base_R_Index1'),
        index2: findBone('mixamorigRightHandIndex2') || findBone('CC_Base_R_Index2'),
        index3: findBone('mixamorigRightHandIndex3') || findBone('CC_Base_R_Index3'),
        index4: findBone('mixamorigRightHandIndex4') || findBone('CC_Base_R_Index3'),

        middle1: findBone('mixamorigRightHandMiddle1') || findBone('CC_Base_R_Mid1'),
        middle2: findBone('mixamorigRightHandMiddle2') || findBone('CC_Base_R_Mid2'),
        middle3: findBone('mixamorigRightHandMiddle3') || findBone('CC_Base_R_Mid3'),
        middle4: findBone('mixamorigRightHandMiddle4') || findBone('CC_Base_R_Mid3'),

        ring1: findBone('mixamorigRightHandRing1') || findBone('CC_Base_R_Ring1'),
        ring2: findBone('mixamorigRightHandRing2') || findBone('CC_Base_R_Ring2'),
        ring3: findBone('mixamorigRightHandRing3') || findBone('CC_Base_R_Ring3'),
        ring4: findBone('mixamorigRightHandRing4') || findBone('CC_Base_R_Ring3'),

        pinky1: findBone('mixamorigRightHandPinky1') || findBone('CC_Base_R_Pinky1'),
        pinky2: findBone('mixamorigRightHandPinky2') || findBone('CC_Base_R_Pinky2'),
        pinky3: findBone('mixamorigRightHandPinky3') || findBone('CC_Base_R_Pinky3'),
        pinky4: findBone('mixamorigRightHandPinky4') || findBone('CC_Base_R_Pinky3'),

        hand: findBone('mixamorigRightHand') || findBone('CC_Base_R_Hand'),
        forearm: findBone('mixamorigRightForeArm') || findBone('CC_Base_R_Forearm'),
        arm: findBone('mixamorigRightArm') || findBone('CC_Base_R_Upperarm')
    };

    // Verificación
    console.log('[DEBUG] Dedos encontrados:', fingers);

    // --- Levantar el brazo ---
    if (fingers.arm) await this.moveBone(fingers.arm.name, null, { x: 1.0 }, 300);
    if (fingers.forearm) await this.moveBone(fingers.forearm.name, null, { x: -0.6 }, 300);
    if (fingers.hand) await this.moveBone(fingers.hand.name, null, { x: 0.4, y: 0.1 }, 300);

    // --- Cerrar los dedos para formar un puño ---
    const foldRotation = 1.0; // Cambiar a positivo para flexión hacia adentro
    const delay = 40;

    const fingerChains = [
        ['index1', 'index2', 'index3', 'index4'],
        ['middle1', 'middle2', 'middle3', 'middle4'],
        ['ring1', 'ring2', 'ring3', 'ring4'],
        ['pinky1', 'pinky2', 'pinky3', 'pinky4']
    ];

    for (const chain of fingerChains) {
        for (const name of chain) {
            const bone = fingers[name];
            if (bone) {
                await this.moveBone(bone.name, null, { x: foldRotation }, 120);
            }
        }
        await new Promise(r => setTimeout(r, delay));
    }

    // --- Ajustar el pulgar encima del puño ---
    if (fingers.thumb1) await this.moveBone(fingers.thumb1.name, null, { x: 0.6, y: 0.4, z: 0.2 }, 200);
    if (fingers.thumb2) await this.moveBone(fingers.thumb2.name, null, { x: 0.5, y: 0.2 }, 200);
    if (fingers.thumb3) await this.moveBone(fingers.thumb3.name, null, { x: 0.4 }, 200);

    // --- Mantener posición por 2 segundos ---
    await new Promise(r => setTimeout(r, 2000));
}

// IMPORTANTE: Orienta la palma/muñeca de la mano derecha según el modo:
// - 'inward': palma hacia la cámara (hola, chau)
// - 'down': dedos hacia abajo (m, n)
// - 'side': vista de perfil (l, k, o, c)
// - 'left': pulgar hacia arriba horizontal (legusta)

    _ccBaseOutwardPalm(startQ) {
        const qY = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI / 2);
        return startQ.clone().multiply(qY);
    }

    orientRightPalm(mode = 'inward', duration = 300) {
        const hand = this.findBoneByContains('righthand')
            || this.findBoneByContains('mixamorigRightHand')
            || this.findBoneByContains('cc_base_r_hand')
            || this.findBoneByContains('r_hand');

        if (!hand) {
            console.warn('[WARN orientRightPalm] No se encontró el hueso de la mano derecha para rotar la palma');
            return Promise.resolve();
        }
        // Si ya está en el modo solicitado, no hacer nada (evita rotaciones acumulativas)
        if (this._palmOrientationMode === mode) {
            return Promise.resolve();
        }
        if (mode !== 'inward' && mode !== 'down' && mode !== 'left' && mode !== 'side' && mode !== 'side_half') {
            return this.animateBoneQuaternion(hand, { x: 0, y: 0, z: 0 }, duration);
        }

        return new Promise(resolve => {
            const startQ = hand.quaternion.clone();
            let targetQ = startQ.clone();

            const handName = (hand.name || '').toLowerCase();
            const isCCBaseHand = handName.includes('cc_base');

            if (mode === 'inward') {
                // Mantener la palma mirando a la cámara
                if (isCCBaseHand) {
                    const qY = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -Math.PI / 2);
                    targetQ = startQ.clone().multiply(qY);
                } else {
                    const qX = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI);
                    const qZ = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI);
                    targetQ = startQ.clone().multiply(qX).multiply(qZ);
                }
            } else if (mode === 'down') {
                // Girar muñeca 180° hacia abajo y mostrar la parte exterior de la mano en rigs CC_Base
                const qZ = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI);
                if (isCCBaseHand) {
                    // Primero dejar el dorso de la mano hacia la cámara, luego inclinar hacia abajo
                    targetQ = this._ccBaseOutwardPalm(startQ).multiply(qZ);
                } else {
                    // Comportamiento original para otros rigs
                    targetQ = startQ.clone().multiply(qZ);
                }
            } else if (mode === 'left') {
                // 90° CCW alrededor de Z: de "arriba" a "izquierda" (pulgar arriba horizontal)
                const qZ = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI / 2);
                targetQ = startQ.clone().multiply(qZ);
            } else if (mode === 'side') {
                const qY = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -Math.PI / 2);
                targetQ = startQ.clone().multiply(qY);
            } else if (mode === 'side_half') {
                const qY = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -Math.PI / 4);
                targetQ = startQ.clone().multiply(qY);
            }

            const start = performance.now();
            const step = (now) => {
                const totalDuration = duration * 2;
                const t = Math.min((now - start) / totalDuration, 1);
                THREE.Quaternion.slerp(startQ, targetQ, hand.quaternion, t);
                hand.updateMatrixWorld(true);
                if (t < 1) requestAnimationFrame(step); else { this._palmOrientationMode = mode; resolve(); }
            };
            requestAnimationFrame(step);
        });
    }

    rotateRightPalmBack(duration = 250) {
        const hand = this.findBoneByContains('righthand')
            || this.findBoneByContains('mixamorigRightHand')
            || this.findBoneByContains('cc_base_r_hand')
            || this.findBoneByContains('r_hand');

        if (!hand) {
            return Promise.resolve();
        }

        return new Promise(resolve => {
            const startQ = hand.quaternion.clone();
            let targetQ = startQ.clone();

            const handName = (hand.name || '').toLowerCase();
            const isCCBaseHand = handName.includes('cc_base');

            if (isCCBaseHand) {
                const baseQ = this._ccBaseOutwardPalm(startQ);
                const extraY = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI / 2);
                targetQ = baseQ.clone().multiply(extraY);
            } else {
                const qY = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI * 1.5);
                targetQ = startQ.clone().multiply(qY);
            }

            const start = performance.now();
            const step = (now) => {
                const t = Math.min((now - start) / duration, 1);
                THREE.Quaternion.slerp(startQ, targetQ, hand.quaternion, t);
                hand.updateMatrixWorld(true);
                if (t < 1) requestAnimationFrame(step); else resolve();
            };
            requestAnimationFrame(step);
        });
    }

    // Rota la muñeca derecha una cantidad configurable de grados alrededor del eje Y.
    // Pensado como un giro "hacia la izquierda" (desde la perspectiva actual de la mano).
    // Ejemplo de uso: await avatar.rotateRightPalmLeftDegrees(45, 250);
    rotateRightPalmLeftDegrees(degrees, duration = 250, comentario="") {
        console.log(comentario);
              console.log(degrees);
        const hand = this.findBoneByContains('righthand')
            || this.findBoneByContains('mixamorigRightHand')
            || this.findBoneByContains('cc_base_r_hand')
            || this.findBoneByContains('r_hand');
        if (!hand) {
            return Promise.resolve();
        }

        return new Promise(resolve => {
            const startQ = hand.quaternion.clone();
            let targetQ = startQ.clone();

            const handName = (hand.name || '').toLowerCase();
            const isCCBaseHand = handName.includes('cc_base');

            // Convertir grados a radianes
            const angleRad = Math.PI / 12;
 
            if (isCCBaseHand) {
                console.log("CC_BASE");
                // Para rigs CC_Base, partir de una palma con dorso hacia la cámara
                const baseQ = this._ccBaseOutwardPalm(startQ);
                const qY = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), angleRad);
                targetQ = baseQ.clone().multiply(qY);
            } else {
                // Para otros rigs, rotar directamente desde la orientación actual
                const qY = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), angleRad);
                targetQ = startQ.clone().multiply(qY);
            }

            const start = performance.now();
            const step = (now) => {
                const t = Math.min((now - start) / duration, 1);
                THREE.Quaternion.slerp(startQ, targetQ, hand.quaternion, t);
                hand.updateMatrixWorld(true);
                if (t < 1) requestAnimationFrame(step); else resolve();
            };
            requestAnimationFrame(step);
        });
    }

    rotateRightPalmLeft90(duration = 250) {
        const hand = this.findBoneByContains('righthand')
            || this.findBoneByContains('mixamorigRightHand')
            || this.findBoneByContains('cc_base_r_hand')
            || this.findBoneByContains('r_hand');
        if (!hand) {
            return Promise.resolve();
        }

        return new Promise(resolve => {
            const startQ = hand.quaternion.clone();
            let targetQ = startQ.clone();

            const handName = (hand.name || '').toLowerCase();
            const isCCBaseHand = handName.includes('cc_base');

            const angleRad = Math.PI / 4;

            if (isCCBaseHand) {
                const baseQ = this._ccBaseOutwardPalm(startQ);
                const qY = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), angleRad);
                targetQ = baseQ.clone().multiply(qY);
            } else {
                const qY = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), angleRad);
                targetQ = startQ.clone().multiply(qY);
            }

            const start = performance.now();
            const step = (now) => {
                const t = Math.min((now - start) / duration, 1);
                THREE.Quaternion.slerp(startQ, targetQ, hand.quaternion, t);
                hand.updateMatrixWorld(true);
                if (t < 1) requestAnimationFrame(step); else resolve();
            };
            requestAnimationFrame(step);
        });
    }


    // Configuración de la interfaz de usuario
    setupUI() {
        // Crear contenedor para controles
        this.controlsContainer = document.createElement('div');
        this.controlsContainer.style.position = 'absolute';
        this.controlsContainer.style.top = '10px';
        this.controlsContainer.style.left = '10px';
        this.controlsContainer.style.zIndex = '100';
        this.controlsContainer.style.display = 'flex';
        this.controlsContainer.style.flexDirection = 'column';
        this.controlsContainer.style.gap = '8px';
        this.container.appendChild(this.controlsContainer);
 
    }

    createButton(text, onClick) {
        const button = document.createElement('button');
        button.textContent = text;
        button.style.padding = '10px 20px';
        button.style.borderRadius = '6px';
        button.style.border = '2px solid #4a90e2';
        button.style.backgroundColor = '#f8f9fa';
        button.style.color = '#333';
        button.style.fontSize = '14px';
        button.style.fontWeight = 'bold';
        button.style.cursor = 'pointer';
        button.style.transition = 'all 0.2s ease';
        button.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)';
        
        // Efecto hover
        button.onmouseover = () => {
            button.style.backgroundColor = '#e9ecef';
            button.style.transform = 'translateY(-2px)';
            button.style.boxShadow = '0 4px 8px rgba(0,0,0,0.15)';
        };
        
        button.onmouseout = () => {
            button.style.backgroundColor = '#f8f9fa';
            button.style.transform = 'translateY(0)';
            button.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)';
        };
        
        button.onclick = (e) => {
            e.target.style.backgroundColor = '#dee2e6';
            setTimeout(() => {
                e.target.style.backgroundColor = '#e9ecef';
            }, 150);
            onClick();
        };
        
        return button;
    }

    
    // Versión suave de applyHandKeypointsFrame que hace transiciones
    async applyHandKeypointsFrameSmooth(frame, transitionDuration = 400) {
        if (!this.avatar || !frame || this.isTransitioning) return;
        
        // Procesar el frame para obtener las rotaciones objetivo
        const targetPose = this.calculateHandPoseFromFrame(frame);
        if (!targetPose) return;
        
        // Si no hay pose actual, aplicar directamente
        if (!this.currentHandPose) {
            this.currentHandPose = targetPose;
            this.applyHandPoseDirectly(targetPose);
            return;
        }
        
        // Realizar transición suave
        this.isTransitioning = true;
        await this.transitionToHandPose(this.currentHandPose, targetPose, transitionDuration);
        this.currentHandPose = targetPose;
        this.isTransitioning = false;
    }

    // Calcula las rotaciones de los dedos basándose en el frame de keypoints
    calculateHandPoseFromFrame(frame) {
        let arr = frame;
        if (Array.isArray(frame) && Array.isArray(frame[0])) {
            arr = frame.flat();
        } else if (Array.isArray(frame) && frame.length === 63) {
            arr = frame;
        } else if (frame && frame.keypoints) {
            const kp = Array.isArray(frame.keypoints[0]) ? frame.keypoints.flat() : frame.keypoints;
            arr = kp;
        }
        
        // Reflejar keypoints
        const reflectedArr = [];
        for (let i = 0; i < arr.length; i += 3) {
            const x = arr[i] || 0;
            const y = arr[i + 1] || 0;
            const z = arr[i + 2] || 0;
            reflectedArr.push(-x);
            reflectedArr.push(y);
            reflectedArr.push(z);
        }
        arr = reflectedArr;
        
        const points = Array.from({ length: 21 }, (_, i) => ({
            x: arr[i * 3] ?? 0,
            y: arr[i * 3 + 1] ?? 0,
            z: arr[i * 3 + 2] ?? 0
        }));
        
        const v = (a, b) => ({ x: b.x - a.x, y: b.y - a.y, z: (b.z ?? 0) - (a.z ?? 0) });
        const len = (u) => Math.hypot(u.x, u.y, u.z);
        const angleBetween = (u, w) => {
            const lu = len(u), lw = len(w);
            if (lu === 0 || lw === 0) return 0;
            let c = (u.x * w.x + u.y * w.y + u.z * w.z) / (lu * lw);
            c = Math.min(1, Math.max(-1, c));
            return Math.acos(c);
        };
        const angleAt = (prevIdx, jointIdx, nextIdx) => {
            const u = v(points[jointIdx], points[prevIdx]);
            const w = v(points[jointIdx], points[nextIdx]);
            const a = angleBetween(u, w);
            return Math.max(0, Math.PI - a);
        };
        const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
        const wrist = 0;
        const f = {
            thumb: [1, 2, 3, 4],
            index: [5, 6, 7, 8],
            middle: [9, 10, 11, 12],
            ring: [13, 14, 15, 16],
            pinky: [17, 18, 19, 20]
        };
        
        // Calcular rotaciones para cada dedo
        const pose = {};
        
        // Dedos normales
        ['index', 'middle', 'ring', 'pinky'].forEach(fingerName => {
            const chain = f[fingerName];
            const [mcp, pip, dip, tip] = chain;
            const mcpFlex = clamp(angleAt(wrist, mcp, pip), 0, 1.4);
            const pipFlex = clamp(angleAt(mcp, pip, dip), 0, 1.6);
            const dipFlex = clamp(angleAt(pip, dip, tip), 0, 1.3);
            
            pose[fingerName] = {
                bone1: mcpFlex,
                bone2: pipFlex,
                bone3: dipFlex
            };
        });
        
        // Pulgar
        const [t1, t2, t3, t4] = f.thumb;
        const thumb_mcp = clamp(angleAt(t1, t2, t3), 0, 1.3);
        const thumb_ip = clamp(angleAt(t2, t3, t4), 0, 1.3);
        const thumb_cmc = clamp(angleAt(wrist, t1, t2), 0, 1.0);
        
        pose.thumb = {
            bone1: thumb_cmc,
            bone2: thumb_mcp,
            bone3: thumb_ip
        };
        
        return pose;
    }
    
    // Aplica una pose directamente sin transición
    applyHandPoseDirectly(pose) {
        if (!this._rightHandBones) {
            const m = {};
            this.avatar.traverse(n => {
                if (n.isBone && typeof n.name === 'string') {
                    if (n.name.startsWith('mixamorigRightHand') || n.name.startsWith('CC_Base_R_')) {
                        m[n.name] = n;
                    }
                }
            });
            this._rightHandBones = {
                thumb1: m['mixamorigRightHandThumb1'] || m['CC_Base_R_Thumb1'] || null,
                thumb2: m['mixamorigRightHandThumb2'] || m['CC_Base_R_Thumb2'] || null,
                thumb3: m['mixamorigRightHandThumb3'] || m['CC_Base_R_Thumb3'] || null,
                index1: m['mixamorigRightHandIndex1'] || m['CC_Base_R_Index1'] || null,
                index2: m['mixamorigRightHandIndex2'] || m['CC_Base_R_Index2'] || null,
                index3: m['mixamorigRightHandIndex3'] || m['CC_Base_R_Index3'] || null,
                middle1: m['mixamorigRightHandMiddle1'] || m['CC_Base_R_Mid1'] || null,
                middle2: m['mixamorigRightHandMiddle2'] || m['CC_Base_R_Mid2'] || null,
                middle3: m['mixamorigRightHandMiddle3'] || m['CC_Base_R_Mid3'] || null,
                ring1: m['mixamorigRightHandRing1'] || m['CC_Base_R_Ring1'] || null,
                ring2: m['mixamorigRightHandRing2'] || m['CC_Base_R_Ring2'] || null,
                ring3: m['mixamorigRightHandRing3'] || m['CC_Base_R_Ring3'] || null,
                pinky1: m['mixamorigRightHandPinky1'] || m['CC_Base_R_Pinky1'] || null,
                pinky2: m['mixamorigRightHandPinky2'] || m['CC_Base_R_Pinky2'] || null,
                pinky3: m['mixamorigRightHandPinky3'] || m['CC_Base_R_Pinky3'] || null
            };
        }
        
        const B = this._rightHandBones;
        const SIGN_FINGERS = 1;
        const SIGN_THUMB_MCP_IP = 1;
        
        // Aplicar dedos normales
        ['index', 'middle', 'ring', 'pinky'].forEach(fingerName => {
            const fingerPose = pose[fingerName];
            if (fingerPose) {
                if (B[fingerName + '1']) B[fingerName + '1'].rotation.x = SIGN_FINGERS * fingerPose.bone1;
                if (B[fingerName + '2']) B[fingerName + '2'].rotation.x = SIGN_FINGERS * fingerPose.bone2;
                if (B[fingerName + '3']) B[fingerName + '3'].rotation.x = SIGN_FINGERS * fingerPose.bone3;
            }
        });
        
        // Aplicar pulgar
        if (pose.thumb) {
        //     if (B.thumb1) B.thumb1.rotation.x = -pose.thumb.bone1;
        //     if (B.thumb2) B.thumb2.rotation.x = SIGN_THUMB_MCP_IP * pose.thumb.bone2;
        //     if (B.thumb3) B.thumb3.rotation.x = SIGN_THUMB_MCP_IP * pose.thumb.bone3;
        }
    }
    
    // Transiciona suavemente entre dos poses
    async transitionToHandPose(fromPose, toPose, duration) {
        return new Promise(resolve => {
            const startTime = performance.now();
            
            const animate = (time) => {
                const rawT = Math.min((time - startTime) / duration, 1);
                const t = this.easeInOutQuad(rawT);
                
                // Interpolar entre las poses
                const interpolatedPose = this.interpolateHandPoses(fromPose, toPose, t);
                this.applyHandPoseDirectly(interpolatedPose);
                
                if (rawT < 1) {
                    requestAnimationFrame(animate);
                } else {
                    resolve();
                }
            };
            
            requestAnimationFrame(animate);
        });
    }
    
    // Interpola entre dos poses de mano
    interpolateHandPoses(fromPose, toPose, t) {
        const interpolatedPose = {};
        
        ['index', 'middle', 'ring', 'pinky', 'thumb'].forEach(fingerName => {
            if (fromPose[fingerName] && toPose[fingerName]) {
                interpolatedPose[fingerName] = {
                    bone1: fromPose[fingerName].bone1 + (toPose[fingerName].bone1 - fromPose[fingerName].bone1) * t,
                    bone2: fromPose[fingerName].bone2 + (toPose[fingerName].bone2 - fromPose[fingerName].bone2) * t,
                    bone3: fromPose[fingerName].bone3 + (toPose[fingerName].bone3 - fromPose[fingerName].bone3) * t
                };
            } else if (toPose[fingerName]) {
                interpolatedPose[fingerName] = toPose[fingerName];
            }
        });
        
        return interpolatedPose;
    }
    applyHandKeypointsFrame(frame) { 
        // Usar la nueva función suave en lugar de la implementación original
        this.applyHandKeypointsFrameSmooth(frame, this.transitionDuration);
    }

    startHandLeftRight(durationMs = 1000, amplitudeRad = 0.25, speedHz = 0.8) {
        // Reutilizar la nueva variante que mueve la mano derecha-izquierda
        return this.startHandRightLeft(durationMs, amplitudeRad, speedHz);
    }

    // Variante sencilla: movimiento derecha-izquierda usando rotación sobre el eje Y de la mano
    // (sin afectar hombro ni antebrazo).
    startHandRightLeft(durationMs = 1000, amplitudeRad = 0.25, speedHz = 0.8) {
        const hand = this.findBoneByContains('righthand')
            || this.findBoneByContains('mixamorigRightHand')
            || this.findBoneByContains('cc_base_r_hand')
            || this.findBoneByContains('r_hand');

        if (!hand) return Promise.resolve();

        const baseHandY = hand.rotation.y;
        const start = performance.now();

        return new Promise(resolve => {
            const tick = (now) => {
                const elapsed = now - start;
                const t = elapsed / 1000;
                const phase = 2 * Math.PI * speedHz * t;
                const a = amplitudeRad;

                // Oscilar la muñeca hacia derecha-izquierda alrededor de Y
                hand.rotation.y = baseHandY + a * Math.sin(phase);
                hand.updateMatrixWorld(true);

                if (elapsed >= durationMs) {
                    // Restaurar rotación original en Y al final del movimiento
                    hand.rotation.y = baseHandY;
                    hand.updateMatrixWorld(true);
                    return resolve();
                }
                requestAnimationFrame(tick);
            };
            requestAnimationFrame(tick);
        });
    }

    moveHandLeftToRight(durationMs = 800, angleRad = 0.3) {
    const hand = this.findBoneByContains('righthand')
        || this.findBoneByContains('mixamorigRightHand')
        || this.findBoneByContains('cc_base_r_hand')
        || this.findBoneByContains('r_hand');

    if (!hand) return Promise.resolve();

    const startZ = hand.rotation.z;
    const targetZ = startZ + angleRad;
    const start = performance.now();

    return new Promise(resolve => {
        const step = (now) => {
            const t = Math.min((now - start) / durationMs, 1);
            hand.rotation.z = startZ + (targetZ - startZ) * t;
            hand.updateMatrixWorld(true);

            if (t < 1) {
                requestAnimationFrame(step);
            } else {
                // Volver suavemente a la posición original
                const backStart = performance.now();
                const backStep = (nowBack) => {
                    const tb = Math.min((nowBack - backStart) / durationMs, 1);
                    hand.rotation.z = targetZ + (startZ - targetZ) * tb;
                    hand.updateMatrixWorld(true);

                    if (tb < 1) {
                        requestAnimationFrame(backStep);
                    } else {
                        resolve();
                    }
                };
                requestAnimationFrame(backStep);
            }
        };
        requestAnimationFrame(step);
    });
}


    _restoreHandLRBase() {
        const s = this._handLRState;
        if (!s) return;
        const { bones, base } = s;
        if (bones && bones.hand) bones.hand.rotation.z = base.handZ;
    }

    stopHandLeftRight() {
        const s = this._handLRState;
        if (!s) return;
        s.cancelled = true;
        if (s.raf) cancelAnimationFrame(s.raf);
        this._restoreHandLRBase();
        this._handLRState = null;
    }

    startHandForwardBack(durationMs = 1000, amplitudeRad = 0.25, speedHz = 0.8) {
        const hand = this.findBoneByContains('righthand')
            || this.findBoneByContains('mixamorigRightHand')
            || this.findBoneByContains('cc_base_r_hand')
            || this.findBoneByContains('r_hand');
        if (!hand) return Promise.resolve();
        this.stopHandForwardBack();
        const state = {
            cancelled: false,
            raf: null,
            bones: { hand },
            base: {
                handY: hand.rotation.y
            }
        };
        this._handFBState = state;
        const start = performance.now();
        return new Promise(resolve => {
            const tick = (now) => {
                if (state.cancelled) {
                    this._restoreHandFBBase();
                    this._handFBState = null;
                    return resolve();
                }
                const elapsed = now - start;
                const t = elapsed / 1000;
                const phase = 2 * Math.PI * speedHz * t;
                const a = amplitudeRad;
                if (state.bones.hand) state.bones.hand.rotation.y = state.base.handY + a * Math.sin(phase);
                if (elapsed >= durationMs) {
                    this._restoreHandFBBase();
                    this._handFBState = null;
                    return resolve();
                }
                state.raf = requestAnimationFrame(tick);
            };
            state.raf = requestAnimationFrame(tick);
        });
    }

    _restoreHandFBBase() {
        const s = this._handFBState;
        if (!s) return;
        const { bones, base } = s;
        if (bones && bones.hand) bones.hand.rotation.y = base.handY;
    }

    stopHandForwardBack() {
        const s = this._handFBState;
        if (!s) return;
        s.cancelled = true;
        if (s.raf) cancelAnimationFrame(s.raf);
        this._restoreHandFBBase();
        this._handFBState = null;
    }

    detectHandMotion(frames) {
        const getArr = (frame) => {
            if (!frame) return [];
            if (Array.isArray(frame) && Array.isArray(frame[0])) return frame.flat();
            if (Array.isArray(frame)) return frame;
            if (frame && frame.keypoints) return Array.isArray(frame.keypoints[0]) ? frame.keypoints.flat() : frame.keypoints;
            return [];
        };

        let last = null;
        let sumDx = 0, sumDz = 0;
        for (const fr of frames) {
            const arr = getArr(fr);
            if (!arr || arr.length < 3) continue;
            const x = arr[0] ?? 0;
            const z = arr[2] ?? 0;
            if (last) {
                sumDx += Math.abs(x - last.x);
                sumDz += Math.abs(z - last.z);
            }
            last = { x, z };
        }
        const eps = 1e-3;
        if (sumDx > sumDz * 1.2 && sumDx > eps) return 'leftRight';
        if (sumDz > sumDx * 1.2 && sumDz > eps) return 'forwardBack';
        return 'leftRight';
    }

    // Alias para compatibilidad con código existente
    async playFrames(frames, fps = 30, moveArm = true) {
        return this.applyHandKeypointsFrames(frames, fps, moveArm);
    }

    // Reproducir múltiples frames de keypoints aplicándolos progresivamente a la mano
    async applyHandKeypointsFrames(frames, fps = 30, moveArm = true) {
        if (!Array.isArray(frames) || frames.length === 0) return;
        this.isApplyingKeypoints = true;

        try {
            // Si hay 10 o menos frames, aplicar cada frame con transición suave
            if (frames.length <= 10) {
                for (const f of frames) {
                    await this.applyHandKeypointsFrameSmooth(f, this.transitionDuration);
                    // Pequeña pausa entre frames para que se aprecie la transición
                    await new Promise(r => setTimeout(r, 200));
                }
                return;
            }

            // Si hay más de 10 frames, detectar el tipo de movimiento y ejecutar la oscilación correspondiente
            const estDelay = Math.max(1, Math.round(1000 / (fps > 0 ? fps : 30)));
            const totalDuration = frames.length * estDelay;
            let motionType = this.detectHandMotion(frames);
            if (this._nextHandMotionMode === 'leftRightOnly' && motionType === 'forwardBack') {
                motionType = 'leftRight';
            }
            this._nextHandMotionMode = null;

            let motionPromise = null;
            if (motionType === 'leftRight') motionPromise = this.startHandLeftRight(totalDuration);
            else if (motionType === 'forwardBack') motionPromise = this.startHandForwardBack(totalDuration);

            const delay = estDelay;
            for (const f of frames) {
                await this.applyHandKeypointsFrameSmooth(f, Math.min(delay, this.transitionDuration));
                await new Promise(r => setTimeout(r, delay));
            }

            if (motionPromise) {
                await motionPromise;
            }
        } finally {
            // Detener únicamente la oscilación que se pudo haber iniciado
            if (this._handLRState) this.stopHandLeftRight();
            if (this._handFBState) this.stopHandForwardBack();
            this.isApplyingKeypoints = false;
        }
    }



    destroy() {
        // Limpiar animaciones
        this.isPlayingSequence = false;
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
        }

        // Limpiar UI
        if (this.controlsContainer && this.container.contains(this.controlsContainer)) {
            this.container.removeChild(this.controlsContainer);
        }

        // Limpiar renderer
        window.removeEventListener('resize', this.onWindowResize);
        if (this.renderer) {
            this.container.removeChild(this.renderer.domElement);
        }
    }
}

// ============================================================
// MÉTODOS DE MANIPULACIÓN DE BRAZO Y MANO
// ============================================================
//
// setArmsRelaxedPose: Pone los brazos en posición relajada junto al cuerpo.
// - Usado al iniciar el avatar y al terminar una seña.
//
// poseRightArmForAMi: Extiende el brazo derecho frente al pecho, posición base para señas.
// - Usado antes de cualquier seña (m, n, hola, etc).
// - skipPose controla si se salta esta pose cuando la siguiente seña usa el mismo modo.
//
// orientRightPalm: Orienta la palma/muñeca según el modo especificado.
// - 'inward': palma hacia la cámara (hola, chau)
// - 'down': dedos hacia abajo (m, n)
// - 'side': vista de perfil (l, k, o, b, c)
// - 'left': pulgar hacia arriba horizontal (legusta)
//
// rotateRightPalmBack: Rota la muñeca para mostrar el dorso hacia la cámara.
// - Usado específicamente para las letras m y n después de orientRightPalm('down').
//
// restoreRightArmPose: Restaura el brazo a su posición anterior o inicial.
// - Usado después de cada seña para volver a la posición relajada.
// - toInitial=true restaura a pose inicial, false a pose previa.
//
// captureInitialArmPose: Guarda la pose inicial del brazo (brazos abajo).
// - Usado antes de una secuencia de señas para recordar posición base.
//
// clearInitialArmPose: Limpia la pose inicial guardada.
// - Usado después de restaurar el brazo a su posición inicial.
//
// resetPalmOrientationMode: Limpia el modo de orientación de palma actual.
// - Usado al terminar una seña para preparar la siguiente.
//
// animateBoneQuaternion: (Helper) Anima la rotación de un hueso usando quaternions.
// - Usado internamente por otros métodos para suavizar rotaciones.
//
// _ccBaseOutwardPalm: (Helper) Rota la palma para rigs CC_Base.
// - Usado internamente por orientRightPalm y rotateRightPalmBack.
//
// startHandLeftRight, stopHandLeftRight: Inicia/detiene oscilación izq-der.
// - Usado en animaciones con muchos frames que requieren este movimiento.
//
// startHandForwardBack, stopHandForwardBack: Inicia/detiene oscilación adelante-atrás.
// - Usado en animaciones con muchos frames que requieren este movimiento.
//
// detectHandMotion: (Helper) Detecta tipo de movimiento en frames de keypoints.
// - Usado para decidir qué oscilación aplicar en animaciones largas.
//
