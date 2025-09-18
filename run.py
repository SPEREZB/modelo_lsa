from app import create_app
import os

app = create_app()

if __name__ == '__main__':
    # Configuración para desarrollo
    debug = os.environ.get('FLASK_DEBUG', '1') == '1'
    port = int(os.environ.get('PORT', 5000))
    
    print("="*50)
    print(f"Iniciando servidor de API en modo {'DEBUG' if debug else 'PRODUCCION'}")
    print(f"URL: http://localhost:{port}")
    print("\nEndpoints disponibles:")
    print("  POST /api/detect    - Detecta puntos clave en una imagen")
    print("  POST /api/predict   - Predice el gesto de la lengua de señas")
    print("  POST /api/reset     - Reinicia la secuencia de predicción")
    print("="*50)
    
    # Iniciar la aplicación
    app.run(host='0.0.0.0', port=port, debug=debug, use_reloader=debug)
