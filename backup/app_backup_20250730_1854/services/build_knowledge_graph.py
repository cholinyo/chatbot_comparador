import os
import pickle
import networkx as nx
from pathlib import Path
import argparse

# Comprobación de modelo spaCy
try:
    import spacy
    nlp = spacy.load("es_core_news_md")
except OSError:
    print("❌ No se pudo cargar el modelo 'es_core_news_md'.")
    print("👉 Instálalo con: python -m spacy download es_core_news_md")
    exit(1)

# Argumentos
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='documents', choices=['documents', 'web', 'apis'],
                    help='Fuente a procesar desde vectorstore/')
args = parser.parse_args()

# Determinar ruta según fuente
VECTORSTORE_PATH = Path(f"vectorstore/{args.source}")
if not VECTORSTORE_PATH.exists():
    raise FileNotFoundError(f"No existe la carpeta vectorstore/{args.source}")

# Cargar fragmentos y metadatos
with open(VECTORSTORE_PATH / "fragmentos.pkl", "rb") as f:
    fragmentos = pickle.load(f)

with open(VECTORSTORE_PATH / "metadatos.pkl", "rb") as f:
    metadatos = pickle.load(f)

# Construir grafo
G = nx.DiGraph()

for i, texto in enumerate(fragmentos):
    doc = nlp(texto)
    entidades = [(ent.text.strip(), ent.label_) for ent in doc.ents if ent.label_ in {"PER", "ORG", "LOC"}]

    for (ent1, tipo1) in entidades:
        G.add_node(ent1, tipo=tipo1)
        for (ent2, tipo2) in entidades:
            if ent1 != ent2:
                G.add_node(ent2, tipo=tipo2)
                G.add_edge(ent1, ent2, fragmento=texto[:200], fuente=metadatos[i].get("fuente", "desconocida"))

# Verificación antes de guardar
if G.number_of_nodes() == 0:
    print(f"⚠️ El grafo para '{args.source}' no contiene nodos. No se guardará.")
    exit(0)

# Guardar grafo
output_dir = Path("graphstore")
output_dir.mkdir(exist_ok=True)

output_base = output_dir / f"knowledge_graph_{args.source}"
nx.write_graphml(G, output_base.with_suffix(".graphml"))

with open(output_base.with_suffix(".gpickle"), "wb") as f:
    pickle.dump(G, f)

print(f"✅ Grafo '{args.source}' generado con {G.number_of_nodes()} nodos y {G.number_of_edges()} relaciones.")