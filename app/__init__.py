from flask import Flask, render_template

def create_app():
    app = Flask(__name__)
    app.secret_key = "clave-secreta-segura"  # 🔐 cámbiala en producción

    # 🔹 Importa y registra blueprints
    from app.routes.chat import chat_bp
    from app.routes.admin import admin_bp
    from app.routes.grafo import grafo_bp
    from app.routes.config import config_bp
    from app.routes.vectorstore import vectorstore_bp
    from app.routes.fragmentos import fragmentos_bp

    app.register_blueprint(chat_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(grafo_bp)
    app.register_blueprint(config_bp)  
    app.register_blueprint(vectorstore_bp)
    app.register_blueprint(fragmentos_bp)

    # ✅ Manejador para error 404 - Página no encontrada
    @app.errorhandler(404)
    def pagina_no_encontrada(error):
        return render_template("error.html", error_code=404, error_message="Página no encontrada"), 404

    # ✅ Manejador para error 500 - Error interno
    @app.errorhandler(500)
    def error_interno(error):
        return render_template("error.html", error_code=500, error_message="Ha ocurrido un error interno"), 500

    return app
