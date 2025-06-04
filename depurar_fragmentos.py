# Guarda este contenido en un archivo por ejemplo llamado: depurar_fragmentos.py
# Luego ejecútalo con: python depurar_fragmentos.py

import pickle

ruta = "vectorstore/documents/fragmentos.pkl"

with open(ruta, "rb") as f:
    fragmentos = pickle.load(f)

print(f"Total fragmentos cargados: {len(fragmentos)}")

# Mostrar los primeros 5 fragmentos
for i, frag in enumerate(fragmentos[:5]):
    print(f"\n--- Fragmento {i+1} ---")
    print(repr(frag))
