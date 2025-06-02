# 🧠 Chatbot Comparador con RAG (OpenAI vs Local LLaMA)

Este proyecto es una aplicación Flask diseñada para comparar respuestas entre dos modelos de lenguaje:

- ✅ Un modelo **local** (LLaMA 3.1 8B Instruct, en formato `.gguf`) usando `llama-cpp-python`
- ☁️ Un modelo **OpenAI** (por ejemplo, `gpt-4` o `gpt-3.5-turbo`)

Además, el sistema permite incorporar **fuentes externas de conocimiento** mediante RAG (Retrieval-Augmented Generation).

---

## ✨ Características principales

### 🔍 Comparador de modelos
- Pregunta al modelo local y a OpenAI simultáneamente
- Visualiza y compara respuestas
- Selección de modelos desde `/admin`

### 🧠 RAG: Configuración de fuentes
Desde la página `/config` puedes configurar múltiples fuentes:

- 📁 **Carpetas de documentos locales** (PDF, DOCX, TXT)
- 🌐 **URLs individuales** con profundidad de indexación
- 🔌 **APIs externas** (con o sin autenticación vía `.env`)
- 🗃️ **Bases de datos** SQL (URI + consulta)

---

## 🛠️ Estructura del proyecto

```
chatbot_comparador/
├── app/
│   ├── routes/           # Blueprints: chat, admin, config, system, grafo
│   ├── services/         # Lógica de comparación e ingestión (FAISS, APIs, etc.)
│   ├── templates/        # HTML con Bootstrap
│   ├── config/           # `settings.json` con fuentes configuradas
├── models/               # Modelos LLaMA `.gguf`
├── logs/                 # Logs de administración
├── .env                  # Variables privadas (API keys)
├── run.py                # Punto de entrada Flask
└── requirements.txt
```

---

## ⚙️ Configuración

### 1. Variables de entorno (`.env`)

Guarda tu clave de OpenAI y claves de API aquí:

```
OPENAI_API_KEY=sk-...
API_KEY_MIS_DATOS=abc123
API_KEY_ONDA=xyz456
```

### 2. Instalación y ejecución

```bash
python -m venv venv
venv\Scripts\activate    # En Windows
pip install -r requirements.txt
python run.py
```

Luego visita: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🔧 Configuración desde el navegador

- `/admin`: seleccionar modelos (local y OpenAI)
- `/config`: añadir fuentes para RAG (carpetas, URLs, APIs, BBDD)

---

## 🚧 Próximas funcionalidades

- Ingestión real de documentos y web
- Vectorización con FAISS y búsqueda semántica
- Integración con grafos de conocimiento
- Panel de control y métricas

---

## 👨‍💻 Autor
Desarrollado por Vicente Caruncho (@vcaruncho) – Ayuntamiento de Onda
