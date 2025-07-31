@echo off
echo Iniciando sistema de chatbots...

REM Activar entorno virtual si existe
if exist "venv\Scripts\activate.bat" (
    echo Activando entorno virtual...
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
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
