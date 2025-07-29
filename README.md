# 🧠 Chatbot Comparador RAG

Este proyecto Flask permite comparar respuestas generadas por dos modelos de lenguaje (uno local y otro de OpenAI) usando RAG (Retrieval-Augmented Generation). Además, ofrece un chat conversacional que accede a fuentes múltiples: documentos, páginas web, APIs y bases de datos. También se pueden construir y visualizar grafos de conocimiento a partir de los datos vectorizados.

---

## 🏗️ Arquitectura General

- Flask para la interfaz web
- Recuperación semántica con FAISS + embeddings
- Modelo de lenguaje local y/o OpenAI
- Ingestión de fuentes:
  - Documentos (PDF, DOCX, TXT)
  - Web (crawler + Selenium)
  - APIs (GET configurables)
  - Bases de datos (próximamente)
- Construcción de grafos de conocimiento (`/grafo`)

---

## 📦 Estructura del Proyecto

```
chatbot_comparador/
├── app/
│   ├── routes/                  # Blueprints Flask (chat, config, comparador, grafo)
│   ├── services/                # Scripts de ingesta (documentos, web, APIs, grafo)
│   ├── utils/                   # Utilidades (carga documentos, RAG, limpieza)
│   ├── templates/               # Plantillas HTML (Jinja2)
│   ├── config/settings.json     # Configuración de fuentes RAG
│   └── static/
├── graphstore/                 # Grafos generados desde distintas fuentes
├── vectorstore/                # Índices FAISS por fuente
│   ├── documents/
│   ├── web/
│   └── apis/
├── run.py                      # Lanzador principal
└── requirements.txt
```

---

## ⚙️ Funcionalidades

### 🔄 Comparador `/comparar`
Realiza una pregunta y muestra 2 respuestas:
- 🧠 Modelo Local
- ☁️ Modelo OpenAI

### 💬 Chat Unificado `/chat`
Consulta a todas las fuentes vectorizadas.
- Recuperación contextual con FAISS
- Solo utiliza el modelo **local** por defecto

### ⚙️ Configuración vía `/config`
Desde esta ruta puedes añadir fuentes:
- 📄 Documentos
- 🌐 URLs
- 🔌 APIs

### 🌐 Grafo de Conocimiento `/grafo`
- Construye grafos a partir de las entidades extraídas de los textos ya indexados
- Visualiza el grafo de cada fuente (`documents`, `web`, `apis`)
- Usa `spaCy` y `networkx` para la extracción y estructuración

---

## ✨ Sistema de Ingestión y Vectorización

- Splitter semántico con `RecursiveCharacterTextSplitter` para fragmentos lógicos
- Limpieza avanzada del texto (`limpiar_texto`)
- Modelo configurable desde `settings.json` (por defecto: `all-MiniLM-L6-v2`)
- Guardado de embeddings con FAISS + metadatos (fuente, origen, etiquetas)

```json
"embedding_model": "all-MiniLM-L6-v2"
```

---

## 🚀 Scripts de Ingestión

```bash
python -m app.services.ingest_documents        # Ingesta de PDF, TXT, DOCX
python -m app.services.ingest_web              # Crawling e indexación web
python -m app.services.ingest_apis             # Conexión a APIs configuradas
python -m app.services.build_knowledge_graph --source documents  # Construcción del grafo desde documentos
python -m app.services.build_knowledge_graph --source web        # Construcción del grafo desde web
```

---

## 🔧 Requisitos e Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/tuusuario/chatbot_comparador.git
cd chatbot_comparador
```

2. Crea un entorno virtual (opcional pero recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instala las dependencias:
```bash
pip install -r requirements.txt
```

Dependencias clave:
- `sentence-transformers`, `faiss-cpu`, `selenium`, `unstructured`
- `langchain`, `spacy`, `networkx` (para el grafo)
- `webdriver-manager` (opcional para Selenium)

**Opcional para mejorar soporte PDF:**
- `pdfplumber`, `PyMuPDF`, o `poppler` (ver instrucciones en `doc_loader.py`)

---

## ▶️ Ejecución

1. Lanza la aplicación Flask:
```bash
python run.py
```

2. Accede en el navegador a:
```
http://127.0.0.1:5000/
```

Desde allí podrás:
- Comparar respuestas de modelos
- Probar el chat contextual
- Configurar fuentes y lanzar ingesta
- Visualizar grafos en `/grafo`

---

## 📋 Futuras mejoras

- Re-rankeo con LLM de fragmentos recuperados
- Ingesta por dominio completo (web crawling profundo)
- Indexado jerárquico por documento
- RAG-fusion entre fuentes
- UI para exploración de fuentes vectorizadas y grafos

---

## 📄 Licencia

MIT
