# 🧰 Módulo `utils`: Carga de documentos

Este módulo contiene las funciones encargadas de **leer, limpiar y preparar documentos** desde múltiples formatos para su posterior vectorización e indexación en el sistema RAG.

---

## 📄 `doc_loader.py`

Este archivo contiene los métodos necesarios para procesar documentos en formatos estándar como:

- `.txt`
- `.pdf` (usando `unstructured`)
- `.docx`
- `.html`

---

## 🔍 Función principal

### `cargar_documentos(rutas_directas=None)`

Devuelve una lista de diccionarios con la estructura:

```python
[
  {
    "nombre": "documento.pdf",
    "texto": "fragmento1\nfragmento2\nfragmento3",
    "origen": "documento"
  },
  ...
]
```

Esto es utilizado por el script `ingest_documents.py`.

---

## 🧩 Otras funciones

| Función         | Descripción                                           |
|-----------------|-------------------------------------------------------|
| `leer_txt()`    | Lee archivos de texto plano.                         |
| `leer_docx()`   | Extrae párrafos de documentos Word.                  |
| `leer_pdf()`    | Usa `unstructured` para dividir PDFs en elementos.   |
| `leer_html()`   | Elimina etiquetas y obtiene texto plano del HTML.    |
| `partir_en_bloques()` | Divide el contenido en fragmentos por longitud.     |

---

## 🛠️ Configuración

El archivo lee `document_folders` desde:

```json
app/config/settings.json
```

Si no se pasa `rutas_directas`, se utiliza esta configuración para buscar documentos.

---

## 📝 Logging

Todos los eventos se registran en:

```
logs/loader.log
```

Incluye advertencias para documentos vacíos, errores de lectura, y resumen final del total procesado.

---

## ✅ Compatibilidad

Este módulo es compatible con:
- `ingest_documents.py`
- `rag_context.py`
- Vectorstore FAISS

---

