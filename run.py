from app_web import create_app
import os

app, socketio = create_app()

if __name__ == '__main__':
    # Configuración para desarrollo
    debug = os.environ.get('FLASK_DEBUG', '1') == '1'
    port = int(os.environ.get('PORT', 5000))
    
    print("="*50)
    print(f"Iniciando servidor de API en modo {'DEBUG' if debug else 'PRODUCCION'}")
    print(f"URL: http://localhost:{port}")
    print("\nEndpoints disponibles:")
    print("  WebSocket /socket.io - Para comunicación en tiempo real")
    print("  GET  /              - Página principal de la aplicación")
    print("  POST /api/detect    - Detecta puntos clave en una imagen")
    print("  POST /api/predict   - Predice el gesto de la lengua de señas")
    print("  POST /api/reset     - Reinicia la secuencia de predicción")
    print("  GET  /api/capture   - Página para capturar nuevas muestras")
    print("="*50)
    
    # Iniciar la aplicación con SocketIO
    try:
        socketio.run(
            app, 
            debug=debug, 
            host='0.0.0.0', 
            port=port,
            allow_unsafe_werkzeug=True,
            use_reloader=debug,
            log_output=debug
        )
    except KeyboardInterrupt:
        print("\nDeteniendo el servidor...")
    except Exception as e:
        print(f"Error al iniciar el servidor: {e}")
        raise
