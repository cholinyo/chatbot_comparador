import logging
from flask import Flask, render_template

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_app():
    app = Flask(__name__)
    app.secret_key = "clave-secreta-segura-tfm-2025"  # 🔐 Cambiar en producción

    # 🔹 Importar y registrar blueprints
    from app.routes.chat import chat_bp
    from app.routes.admin import admin_bp
    from app.routes.grafo import grafo_bp
    from app.routes.config import config_bp
    from app.routes.vectorstore import vectorstore_bp
    from app.routes.fragmentos import fragmentos_bp
    from app.routes.comparador import comparador_bp

    # Registrar blueprints
    app.register_blueprint(chat_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(grafo_bp)
    app.register_blueprint(config_bp)  
    app.register_blueprint(vectorstore_bp)
    app.register_blueprint(fragmentos_bp)
    app.register_blueprint(comparador_bp)

    # Ruta principal
    @app.route("/")
    def index():
        return render_template("index.html")

    # ✅ Manejadores de error
    @app.errorhandler(404)
    def pagina_no_encontrada(error):
        return render_template("error.html", 
                             error_code=404, 
                             error_message="Página no encontrada"), 404

    @app.errorhandler(500)
    def error_interno(error):
        return render_template("error.html", 
                             error_code=500, 
                             error_message="Ha ocurrido un error interno del servidor"), 500

    # Información del sistema en contexto global
    @app.context_processor
    def inject_system_info():
        from app.services.model_manager import model_manager
        
        try:
            status = model_manager.get_system_status()
            return {
                'system_status': status,
                'local_available': status['local']['ollama_available'] or status['local']['file_model_available'],
                'openai_available': status['openai']['configured']
            }
        except Exception as e:
            logging.error(f"Error obteniendo estado del sistema: {e}")
            return {
                'system_status': None,
                'local_available': False,
                'openai_available': False
            }

    return app