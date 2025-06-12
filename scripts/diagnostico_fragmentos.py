import pickle
import spacy
from pathlib import Path

fuente = "web"  # Cambia a 'documents' o 'apis' si quieres revisar otra fuente
ruta = Path(f"vectorstore/{fuente}/fragmentos.pkl")

if not ruta.exists():
    print(f"❌ No se encontró el archivo: {ruta}")
    exit(1)

with open(ruta, "rb") as f:
    fragmentos = pickle.load(f)

print(f"📄 Total de fragmentos cargados: {len(fragmentos)}")

try:
    nlp = spacy.load("es_core_news_md")
except OSError:
    print("""❌ spaCy no pudo cargar el modelo 'es_core_news_md'.
👉 Ejecuta: python -m spacy download es_core_news_md""")
    exit(1)

total_entidades = 0
for i, frag in enumerate(fragmentos[:10]):
    doc = nlp(frag)
    entidades = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in {"PER", "ORG", "LOC"}]
    print(f"🧠 Fragmento {i + 1}: {len(entidades)} entidades -> {entidades}")
    total_entidades += len(entidades)

print(f"✅ Entidades totales detectadas en los primeros 10 fragmentos: {total_entidades}")