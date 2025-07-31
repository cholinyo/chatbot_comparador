#!/bin/bash
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
