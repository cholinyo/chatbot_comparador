# 🧠 Prototipo de Chatbot Interno para Administraciones Locales

> **Trabajo Final de Máster - Sistemas Inteligentes**  
> Comparación entre Modelos de Lenguaje Locales vs OpenAI usando RAG

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-purple.svg)](https://langchain.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Descripción

Este proyecto implementa un **chatbot conversacional interno** diseñado específicamente para administraciones locales, que permite al personal técnico municipal consultar información institucional mediante lenguaje natural. El sistema compara el rendimiento entre **modelos de lenguaje locales** (usando Ollama) y **servicios comerciales** (OpenAI) mediante una arquitectura **RAG (Retrieval-Augmented Generation)**.

### 🎯 Objetivos del TFM

- **Objetivo Principal**: Evaluar la viabilidad de modelos locales vs servicios comerciales en entornos de administración pública
- **Soberanía Tecnológica**: Implementar soluciones que cumplan con el CCN-TEC 014 y el Esquema Nacional de Seguridad
- **Arquitectura RAG**: Integrar múltiples fuentes de conocimiento (documentos, APIs, bases de datos, web)
- **Comparación Empírica**: Analizar métricas de rendimiento, latencia, costes y calidad de respuestas

---

## 🏗️ Arquitectura del Sistema

```mermaid
graph TB
    A[Usuario] --> B[Flask Web App]
    B --> C[Comparador de Modelos]
    C --> D[LangChain OpenAI]
    C --> E[LangChain Ollama]
    
    F[Sistema RAG] --> G[Vector Store FAISS]
    F --> H[Ingesta Multi-fuente]
    
    H --> I[📄 Documentos]
    H --> J[🌐 URLs]
    H --> K[🔌 APIs]
    H --> L[🗃️ Bases de Datos]
    
    G --> C
    
    subgraph "Modelos Locales"
        E --> M[Llama 3.1]
        E --> N[Mistral 7B]
        E --> O[Gemma]
    end
    
    subgraph "Servicios Comerciales"
        D --> P[GPT-4]
        D --> Q[GPT-3.5]
    end
```

### 🔧 Componentes Clave

1. **Frontend Web**: Interfaz Flask con Bootstrap para comparación de modelos
2. **Backend RAG**: Sistema de recuperación aumentada con LangChain
3. **Vector Store**: FAISS para búsqueda semántica eficiente
4. **Ingesta Multi-fuente**: Pipelines para documentos, APIs, web y BBDD
5. **Comparador de Modelos**: Evaluación lado a lado con métricas detalladas

---

## 🚀 Instalación y Configuración

### Prerrequisitos

- **Python 3.9+**
- **Git**
- **Ollama** (para modelos locales)
- **OpenAI API Key** (opcional, para comparación)

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/chatbot_comparador.git
cd chatbot_comparador
```

### 2. Crear Entorno Virtual

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar Variables de Entorno

Crear archivo `.env` en la raíz del proyecto:

```env
# OpenAI (opcional)
OPENAI_API_KEY=sk-your-key-here

# Configuración Flask
FLASK_ENV=development
FLASK_DEBUG=True

# Base de datos (futuro)
DATABASE_URL=sqlite:///data.db
```

### 5. Instalar y Configurar Ollama

```bash
# Instalar Ollama (Windows)
winget install Ollama.Ollama

# Iniciar servidor Ollama
ollama serve

# Descargar modelos (en otra terminal)
ollama pull llama3.1
ollama pull mistral
ollama pull gemma:7b
```

### 6. Inicializar Vector Store

```bash
# Crear directorios necesarios
mkdir -p vectorstore/{documents,web,apis,bbdd}
mkdir -p logs

# Ejecutar ingesta inicial (opcional)
python -m app.services.ingest_documents
```

---

## 💡 Uso del Sistema

### Iniciar la Aplicación

```bash
# Desarrollo
python run.py

# O usando Flask directamente
flask run --host=0.0.0.0 --port=5000
```

La aplicación estará disponible en: `http://localhost:5000`

### Navegación Principal

- **`/chat`** - Chat unificado con RAG
- **`/admin`** - Administración de modelos
- **`/admin/model-tuning`** - Configuración avanzada de parámetros
- **`/config`** - Configuración de fuentes RAG
- **`/vectorstore`** - Estado del vector store
- **`/comparar`** - Comparación lado a lado de modelos

### Configurar Fuentes de Conocimiento

1. **Documentos**: Añadir carpetas con PDFs, DOCXs, TXTs en `/config`
2. **URLs**: Configurar crawling de sitios web institucionales
3. **APIs**: Integrar APIs municipales o gubernamentales
4. **Bases de Datos**: Conectar con sistemas SQL existentes

---

## 📊 Estructura del Proyecto

```
chatbot_comparador/
├── app/
│   ├── __init__.py              # Factory Flask
│   ├── routes/                  # Blueprints
│   │   ├── chat.py             # Chat unificado
│   │   ├── admin.py            # Administración
│   │   ├── config.py           # Configuración RAG
│   │   ├── comparador.py       # Comparación modelos
│   │   └── vectorstore.py      # Gestión vectorstore
│   ├── services/               # Lógica de negocio
│   │   ├── bot_openai.py       # Servicio OpenAI + LangChain
│   │   ├── bot_local.py        # Servicio Ollama + LangChain
│   │   ├── ingest_documents.py # Ingesta documentos
│   │   ├── ingest_web.py       # Ingesta web
│   │   └── ingest_api.py       # Ingesta APIs
│   ├── utils/                  # Utilidades
│   │   ├── rag_utils.py        # Funciones RAG
│   │   └── doc_loader.py       # Carga documentos
│   ├── templates/              # Plantillas HTML
│   │   ├── base.html
│   │   ├── chat.html
│   │   ├── admin.html
│   │   ├── model_tuning.html
│   │   └── comparar.html
│   └── config/
│       └── settings.json       # Configuración sistema
├── vectorstore/                # Índices FAISS
│   ├── documents/
│   ├── web/
│   ├── apis/
│   └── bbdd/
├── logs/                       # Logs del sistema
├── scripts/                    # Scripts utilitarios
├── requirements.txt            # Dependencias Python
├── run.py                     # Lanzador principal
└── README.md                  # Este archivo
```

---

## 🧪 Evaluación y Métricas

### Métricas Implementadas

1. **Rendimiento**:
   - Latencia de respuesta (segundos)
   - Tokens procesados por segundo
   - Uso de memoria y CPU

2. **Calidad**:
   - Relevancia de fragmentos recuperados
   - Coherencia de respuestas generadas
   - Trazabilidad de fuentes

3. **Costes**:
   - Coste por consulta (OpenAI)
   - Recursos computacionales (modelos locales)
   - Escalabilidad económica

### Casos de Uso Evaluados

- **Consultas normativas**: "¿Qué ordenanza regula los ruidos en el municipio?"
- **Procedimientos administrativos**: "¿Cómo solicitar una licencia de obra?"
- **Información municipal**: "¿Dónde están las cámaras de tráfico?"
- **Tramitación electrónica**: "¿Cómo consulto el estado de mi expediente?"

---

## 🔧 Desarrollo y Contribución

### Estructura de Ramas

- `main` - Versión estable
- `develop` - Desarrollo activo
- `feature/*` - Nuevas funcionalidades
- `fix/*` - Correcciones de errores

### Scripts Útiles

```bash
# Reindexar fuentes RAG
python -m app.services.ingest_documents
python -m app.services.ingest_web
python -m app.services.ingest_api

# Tests (futuro)
python -m pytest tests/

# Formateo de código
black app/
flake8 app/
```

### Añadir Nuevos Modelos

1. **Modelo Local**: Agregar en `bot_local.py` y descargar con `ollama pull`
2. **Modelo Comercial**: Actualizar `bot_openai.py` con nueva configuración
3. **Configuración**: Añadir parámetros en `/admin/model-tuning`

---

## 📈 Resultados del TFM

### Hallazgos Principales

1. **Latencia**: Los modelos locales tienen latencia variable (1-10s) vs OpenAI consistente (1-3s)
2. **Calidad**: GPT-4 superior en comprensión compleja, modelos locales competitivos en tareas específicas
3. **Costes**: Modelos locales coste fijo inicial, OpenAI coste variable por uso
4. **Privacidad**: Modelos locales cumplen mejor con requisitos de soberanía de datos

### Recomendaciones para Administraciones

- **Uso híbrido**: Modelos locales para consultas frecuentes, OpenAI para casos complejos
- **Implementación gradual**: Comenzar con modelos locales pequeños, escalar según necesidades
- **Governance**: Establecer políticas claras de uso y supervisión humana

---

## 🛡️ Seguridad y Cumplimiento

### Cumplimiento Normativo

- **ENS (Esquema Nacional de Seguridad)**: Separación de módulos, logs de auditoría
- **CCN-TEC 014**: Recomendaciones para sistemas con IA
- **RGPD**: Protección de datos personales, anonimización
- **Transparencia**: Trazabilidad de respuestas y fuentes

### Medidas de Seguridad

- Autenticación de usuarios (futuro)
- Cifrado de comunicaciones
- Logs de auditoría completos
- Validación de entradas
- Rate limiting API

---

## 🚀 Despliegue en Producción

### Opciones de Despliegue

1. **Docker** (recomendado):
```bash
# Crear imagen
docker build -t chatbot-comparador .

# Ejecutar contenedor
docker run -p 5000:5000 chatbot-comparador
```

2. **Cloud Platforms**:
   - **Azure**: App Service + Container Instance
   - **AWS**: EC2 + ECS
   - **Google Cloud**: Cloud Run + Compute Engine

3. **On-Premise**:
   - Servidor Ubuntu/CentOS
   - Nginx como proxy reverso
   - Systemd para gestión de servicios

### Variables de Entorno Producción

```env
FLASK_ENV=production
FLASK_DEBUG=False
OPENAI_API_KEY=sk-prod-key
DATABASE_URL=postgresql://user:pass@host:5432/db
OLLAMA_HOST=http://ollama-server:11434
```

---

## 📚 Documentación Adicional

### Artículos Relacionados

- [Guía CCN-TEC 014 para IA en AAPP](https://www.ccn-cert.cni.es/)
- [Estrategia Nacional de IA 2024](https://portal.mineco.gob.es/)
- [LangChain Documentation](https://python.langchain.com/)
- [Ollama Model Library](https://ollama.ai/library)

### Papers de Referencia

1. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (Lewis et al., 2020)
2. *LLaMA: Open and Efficient Foundation Language Models* (Touvron et al., 2023)
3. *GPT-4 Technical Report* (OpenAI, 2023)

---

## 🐛 Problemas Conocidos

### Issues Comunes

1. **Ollama no responde**: Verificar que `ollama serve` esté ejecutándose
2. **OpenAI rate limit**: Implementar rate limiting en código
3. **Memoria insuficiente**: Modelos locales requieren 8GB+ RAM
4. **Lentitud en ingesta**: Procesar documentos grandes en lotes

### Soluciones

```bash
# Verificar estado Ollama
curl http://localhost:11434/api/tags

# Limpiar vector store
rm -rf vectorstore/*/
python -m app.services.ingest_documents

# Verificar logs
tail -f logs/admin.log
```

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crear rama feature (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir Pull Request

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

---

## 👨‍💻 Autor

**Vicente Caruncho Ramos**  
📧 Email: -
🎓 Máster en Sistemas Inteligentes - Universitat Jaume I  
📅 Curso: 2024-2025

---

## 🙏 Agradecimientos

- **Tutor**: -
- **LangChain Community** por el framework
- **Ollama Team** por simplificar modelos locales
- **OpenAI** por proporcionar API de comparación
- **Comunidad Flask** por la base web

---

## 📊 Estado del Proyecto

- ✅ **Fase 1**: Arquitectura base y comparador básico
- ✅ **Fase 2**: Integración LangChain y RAG
- 🔄 **Fase 3**: Evaluación empírica y métricas (en progreso)
- ⏳ **Fase 4**: Documentación final TFM
- ⏳ **Fase 5**: Despliegue en cloud para demostraciones

---

## 📞 Soporte

Si encuentras problemas o tienes preguntas:

1. **Issues**: Abre un issue en GitHub
2. **Documentación**: Consulta este README
3. **Logs**: Revisa archivos en `/logs`
4. **Comunidad**: Participa en discusiones del repositorio

---

*Última actualización: Julio 2025*