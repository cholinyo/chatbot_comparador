
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de migración simplificado para Windows
Evita problemas de encoding con emojis
"""

import os
import sys
import json
import shutil
import logging
from pathlib import Path

# Configurar logging sin caracteres especiales
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('migration.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Verificar dependencias necesarias"""
    logger.info("Verificando dependencias...")
    
    dependencies = {
        'llama_cpp': 'llama-cpp-python',
        'requests': 'requests', 
        'flask': 'flask'
    }
    
    missing = []
    for module, package in dependencies.items():
        try:
            __import__(module)
            logger.info(f"OK: {package}")
        except ImportError:
            logger.error(f"FALTA: {package}")
            missing.append(package)
    
    if missing:
        logger.error(f"Instalar dependencias faltantes: pip install {' '.join(missing)}")
        return False
    
    return True

def check_ollama():
    """Verificar Ollama"""
    logger.info("Verificando Ollama...")
    
    ollama_exe = shutil.which("ollama")
    if ollama_exe:
        logger.info(f"Ollama encontrado: {ollama_exe}")
        
        try:
            import subprocess
            result = subprocess.run([ollama_exe, "--version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"Version: {result.stdout.strip()}")
                
                # Test de servicio
                try:
                    import requests
                    response = requests.get("http://localhost:11434/api/tags", timeout=5)
                    if response.status_code == 200:
                        models = response.json().get("models", [])
                        logger.info(f"Servicio activo - {len(models)} modelos")
                        return True
                except:
                    logger.warning("Servicio no responde")
        except Exception as e:
            logger.warning(f"Error verificando Ollama: {e}")
    else:
        logger.warning("Ollama no encontrado - Descargar de: https://ollama.ai/")
    
    return False

def scan_models():
    """Escanear modelos GGUF"""
    logger.info("Escaneando modelos GGUF...")
    
    models_dir = Path("models")
    if not models_dir.exists():
        logger.info("Creando directorio models/")
        models_dir.mkdir(exist_ok=True)
        return []
    
    gguf_files = []
    for gguf_file in models_dir.rglob("*.gguf"):
        size_mb = gguf_file.stat().st_size / (1024 * 1024)
        gguf_files.append({
            "path": str(gguf_file.relative_to(models_dir)),
            "size_gb": round(size_mb / 1024, 1)
        })
        logger.info(f"Encontrado: {gguf_file.name} ({round(size_mb/1024, 1)} GB)")
    
    logger.info(f"Total archivos GGUF: {len(gguf_files)}")
    return gguf_files

def migrate_config():
    """Migrar configuración"""
    logger.info("Migrando configuración...")
    
    config_path = Path("app/config/settings.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configuración por defecto
    default_config = {
        "modelo_local": "",
        "modelo_openai": "gpt-4o",
        "rag_k": 5,
        "document_folders": [],
        "web_sources": [],
        "api_sources": [],
        "db_sources": [],
        "openai_params": {
            "temperature": 0.7,
            "max_tokens": 1024,
            "top_p": 1.0
        },
        "local_params": {
            "temperature": 0.3,
            "max_tokens": 1024,
            "top_k": 40,
            "top_p": 0.7,
            "n_ctx": 2048,
            "n_threads": 6,
            "n_gpu_layers": 0
        },
        "system_settings": {
            "max_concurrent_requests": 5,
            "request_timeout": 120,
            "enable_logging": True,
            "log_level": "INFO",
            "auto_cleanup_models": True
        }
    }
    
    # Cargar configuración existente si existe
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                existing_config = json.load(f)
            logger.info("Configuración existente cargada")
            
            # Backup
            backup_path = config_path.with_suffix(".backup.json")
            shutil.copy2(config_path, backup_path)
            logger.info(f"Backup creado: {backup_path}")
            
            # Migrar valores
            for key, value in existing_config.items():
                if key in default_config:
                    default_config[key] = value
            
        except Exception as e:
            logger.error(f"Error cargando configuración: {e}")
    
    # Guardar configuración
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        logger.info("Configuración migrada correctamente")
        return True
    except Exception as e:
        logger.error(f"Error guardando configuración: {e}")
        return False

def check_environment():
    """Verificar entorno"""
    logger.info("Verificando variables de entorno...")
    
    env_file = Path(".env")
    if env_file.exists():
        logger.info("Archivo .env encontrado")
        try:
            with open(env_file, encoding="utf-8") as f:
                content = f.read()
                if "OPENAI_API_KEY" in content:
                    logger.info("OPENAI_API_KEY configurada")
                else:
                    logger.warning("OPENAI_API_KEY no encontrada")
        except Exception as e:
            logger.error(f"Error leyendo .env: {e}")
    else:
        logger.warning("Archivo .env no encontrado")
        logger.info("Crear .env con: OPENAI_API_KEY=tu_clave_aqui")

def create_scripts():
    """Crear scripts de inicio"""
    logger.info("Creando scripts de inicio...")
    
    # Script Windows
    windows_script = """@echo off
echo Iniciando sistema de chatbots...

REM Activar entorno virtual si existe
if exist "venv\\Scripts\\activate.bat" (
    echo Activando entorno virtual...
    call venv\\Scripts\\activate.bat
) else if exist ".venv\\Scripts\\activate.bat" (
    call .venv\\Scripts\\activate.bat
)

REM Verificar dependencias
echo Verificando dependencias...
python -c "import llama_cpp, flask, requests" || (
    echo Dependencias faltantes. Instalar con:
    echo pip install -r requirements.txt
    pause
    exit /b 1
)

REM Verificar Ollama
where ollama >nul 2>nul
if %errorlevel%==0 (
    echo Ollama disponible
) else (
    echo Ollama no encontrado. Algunos modelos no estaran disponibles.
)

REM Iniciar aplicación
echo Iniciando servidor Flask...
python run.py
pause
"""
    
    # Script Unix
    unix_script = """#!/bin/bash
echo "Iniciando sistema de chatbots..."

# Activar entorno virtual
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Activando entorno virtual..."
    source venv/bin/activate 2>/dev/null || source .venv/bin/activate 2>/dev/null
fi

# Verificar dependencias
echo "Verificando dependencias..."
python -c "import llama_cpp, flask, requests" || {
    echo "Dependencias faltantes. Instalar con:"
    echo "pip install -r requirements.txt"
    exit 1
}

# Verificar Ollama
if command -v ollama &> /dev/null; then
    echo "Ollama disponible"
else
    echo "Ollama no encontrado. Algunos modelos no estaran disponibles."
fi

# Iniciar aplicación
echo "Iniciando servidor Flask..."
python run.py
"""
    
    try:
        # Crear script Windows
        with open("start.bat", "w", encoding="utf-8") as f:
            f.write(windows_script)
        logger.info("Creado: start.bat")
        
        # Crear script Unix
        with open("start.sh", "w", encoding="utf-8") as f:
            f.write(unix_script)
        
        # Hacer ejecutable en Unix
        if os.name != 'nt':
            os.chmod("start.sh", 0o755)
        
        logger.info("Creado: start.sh")
        return True
        
    except Exception as e:
        logger.error(f"Error creando scripts: {e}")
        return False

def test_model_loading():
    """Test opcional de modelos"""
    logger.info("Probando carga de modelos...")
    
    try:
        # Test básico de llama-cpp-python
        from llama_cpp import Llama
        logger.info("llama-cpp-python importado correctamente")
        
        # Test Ollama si está disponible
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                if models:
                    logger.info(f"Ollama tiene {len(models)} modelos disponibles")
                else:
                    logger.info("Ollama funciona pero no tiene modelos")
            else:
                logger.warning("Ollama no responde")
        except Exception as e:
            logger.info(f"Ollama no disponible: {e}")
        
        return True
        
    except Exception as e:
        logger.warning(f"Error en test de modelos: {e}")
        return False

def print_summary():
    """Mostrar resumen final"""
    print("\n" + "="*60)
    print("RESUMEN DE MIGRACIÓN")
    print("="*60)
    print("1. Dependencias verificadas")
    print("2. Configuración migrada")
    print("3. Scripts de inicio creados")
    print("4. Sistema listo para usar")
    print("\nPARA CONTINUAR:")
    print("1. Instalar Ollama (opcional): https://ollama.ai/")
    print("2. Descargar modelos: ollama pull llama3.2:3b")
    print("3. Iniciar aplicación: python run.py")
    print("4. Abrir navegador: http://localhost:5000/admin")
    print("="*60)

def main():
    """Función principal simplificada"""
    print("Iniciando migración del sistema...")
    
    success_count = 0
    total_steps = 5
    
    # 1. Verificar dependencias
    if check_dependencies():
        success_count += 1
        print("✓ Dependencias OK")
    else:
        print("✗ Faltan dependencias")
    
    # 2. Verificar Ollama
    if check_ollama():
        success_count += 1
        print("✓ Ollama OK")
    else:
        print("! Ollama no disponible (opcional)")
    
    # 3. Escanear modelos
    gguf_models = scan_models()
    if len(gguf_models) > 0:
        success_count += 1
        print(f"✓ Modelos GGUF encontrados: {len(gguf_models)}")
    else:
        print("! No hay modelos GGUF (se pueden añadir después)")
    
    # 4. Migrar configuración
    if migrate_config():
        success_count += 1
        print("✓ Configuración migrada")
    else:
        print("✗ Error en configuración")
    
    # 5. Crear scripts
    if create_scripts():
        success_count += 1
        print("✓ Scripts creados")
    else:
        print("✗ Error creando scripts")
    
    # Verificar entorno
    check_environment()
    
    # Test opcional
    test_model_loading()
    
    # Mostrar resumen
    print(f"\nMigración completada: {success_count}/{total_steps} pasos exitosos")
    
    if success_count >= 3:
        print("ÉXITO: Sistema listo para usar")
        print_summary()
    else:
        print("ATENCIÓN: Revisar errores arriba")
        print("Algunos componentes requieren configuración manual")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMigración cancelada por el usuario")
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        print(f"Error inesperado: {e}")
    finally:
        input("\nPresiona Enter para salir...")