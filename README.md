# 🧠 Chatbot Comparador RAG

Este proyecto Flask permite comparar respuestas generadas por dos modelos de lenguaje (uno local y otro de OpenAI) usando RAG (Retrieval-Augmented Generation). Además, ofrece un chat conversacional que accede a fuentes múltiples: documentos, páginas web, APIs y bases de datos.

---

## 📦 Estructura del Proyecto

```
chatbot_comparador/
├── app/
│   ├── routes/                  # Blueprints Flask (chat, config, comparador)
│   ├── services/                # Ingestores e interfaces con modelos
│   ├── utils/                   # Utilidades (RAG, validaciones, etc)
│   ├── templates/               # Plantillas HTML (Jinja2)
│   ├── config/settings.json     # Configuración de fuentes RAG
│   └── static/
├── vectorstore/
│   ├── documents/               # Índices FAISS de documentos
│   ├── web/                     # Índices FAISS de URLs
│   ├── apis/                    # Índices FAISS de APIs
│   └── bbdd/                    # Índices FAISS de bases de datos
└── run.py                       # Lanzador principal
```

---

## ⚙️ Funcionalidades

### 🔄 Comparador
Accede a `/comparar` y realiza una pregunta. Obtendrás dos respuestas:
- 🧠 Modelo Local (por defecto)
- ☁️ OpenAI (si está disponible)

Cada respuesta se basa en la recuperación contextual usando FAISS.

### 💬 Chat Unificado
Accede a `/chat` para preguntar sobre cualquier fuente indexada:
- Recupera fragmentos por relevancia desde documentos, URLs, APIs y bases de datos.
- Se muestra la fuente, la distancia FAISS y el contenido.
- Solo se usa el modelo **local** para generar la respuesta.

### 🧠 Ingestión RAG por fuentes

Puedes configurar y lanzar la ingestión desde:
```
http://127.0.0.1:5000/config
```

Allí puedes añadir:
- 📄 Carpetas de documentos (PDF, DOCX, TXT)
- 🌐 URLs individuales (con profundidad)
- 🔌 APIs configurables (con o sin autenticación)
- 🗃️ Consultas a bases de datos (SQLite u otras)

### ✨ Sistema de vectorización

- Se utiliza `SentenceTransformers` con `all-MiniLM-L6-v2`
- Se crean índices independientes para cada tipo de fuente
- Cada fuente se guarda en su carpeta dentro de `vectorstore/`
- Se utilizan archivos:
  - `index.faiss`
  - `fragmentos.pkl`

---

## 🚫 Clave OpenAI no configurada

Actualmente, el sistema **funciona con modelo local por defecto**.  
Si deseas usar OpenAI, añade tu clave en un archivo `.env` como:

```bash
OPENAI_API_KEY=sk-xxx
```

---

## 🧪 Scripts de ingestión

Puedes ejecutar directamente:

```bash
python -m app.services.ingest_documents
python -m app.services.ingest_web
# futuros: ingest_api.py, ingest_bbdd.py
```

---

## 📋 Requisitos

- Python 3.9+
- Selenium
- FAISS
- SentenceTransformers
- Flask
- webdriver-manager
- unstructured (para PDF)

Instalación recomendada:

```bash
pip install -r requirements.txt
```

---

## 🛠️ Pendiente / Futuro

- Ingesta automatizada por dominio completo
- Historial de preguntas/respuestas
- Interfaz para exploración del vectorstore
- Migración completa a uso de modelos locales y fallback a OpenAI si se desea

---

## 📄 Licencia

MIT