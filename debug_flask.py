#!/usr/bin/env python3
"""
debug_flask.py - Script para diagnosticar problemas en la aplicación Flask
Crear este archivo en la raíz del proyecto
"""

import sys
import os
import traceback

print("🔍 DIAGNÓSTICO DE LA APLICACIÓN FLASK")
print("=" * 50)

# 1. Verificar imports básicos
print("\n1. Verificando imports básicos...")
try:
    from flask import Flask
    print("✅ Flask importado correctamente")
except ImportError as e:
    print(f"❌ Error importando Flask: {e}")
    sys.exit(1)

# 2. Verificar estructura del proyecto
print("\n2. Verificando estructura del proyecto...")
required_files = [
    "app/__init__.py",
    "app/routes/chat.py",
    "app/utils/rag_utils.py",
    "run.py"
]

for file_path in required_files:
    if os.path.exists(file_path):
        print(f"✅ {file_path}")
    else:
        print(f"❌ {file_path} no encontrado")

# 3. Verificar imports de la aplicación
print("\n3. Verificando imports de la aplicación...")

try:
    from app import create_app
    print("✅ create_app importado correctamente")
except ImportError as e:
    print(f"❌ Error importando create_app: {e}")
    print(f"   Traceback: {traceback.format_exc()}")

# 4. Verificar imports específicos de RAG
print("\n4. Verificando imports RAG...")

imports_to_test = [
    ("app.utils.rag_utils", "buscar_fragmentos_combinados"),
    ("app.utils.chroma_store", "get_chroma_store"),
    ("app.services.bot_local", "get_local_response"),
    ("app.services.bot_openai", "get_openai_response"),
]

for module_name, function_name in imports_to_test:
    try:
        module = __import__(module_name, fromlist=[function_name])
        func = getattr(module, function_name)
        print(f"✅ {module_name}.{function_name}")
    except ImportError as e:
        print(f"❌ Error importando {module_name}.{function_name}: {e}")
    except AttributeError as e:
        print(f"❌ Función {function_name} no encontrada en {module_name}: {e}")
    except Exception as e:
        print(f"❌ Error inesperado con {module_name}.{function_name}: {e}")

# 5. Intentar crear la aplicación Flask
print("\n5. Intentando crear aplicación Flask...")
try:
    from app import create_app
    app = create_app()
    print("✅ Aplicación Flask creada correctamente")
    
    # Listar rutas registradas
    print("\n📋 Rutas registradas:")
    with app.app_context():
        for rule in app.url_map.iter_rules():
            print(f"   {rule.methods} {rule.rule}")
            
except Exception as e:
    print(f"❌ Error creando aplicación Flask: {e}")
    print(f"   Traceback completo:")
    print(traceback.format_exc())

# 6. Verificar dependencias críticas
print("\n6. Verificando dependencias críticas...")
critical_packages = [
    "flask",
    "chromadb", 
    "llama_cpp",
    "sentence_transformers",
    "openai",
    "langchain_chroma",
    "langchain_community",
    "llama_index"
]

for package in critical_packages:
    try:
        __import__(package)
        print(f"✅ {package}")
    except ImportError:
        print(f"❌ {package} no instalado")

print("\n" + "=" * 50)
print("🏁 Diagnóstico completado")
print("\nSi ves errores ❌ arriba, esas son probablemente las causas del error 500.")
print("Instala los paquetes faltantes con:")
print("pip install [nombre_paquete]")