from flask import Blueprint, request, render_template, redirect, url_for
import networkx as nx
from pyvis.network import Network
from pathlib import Path
import subprocess
import pickle

graph_bp = Blueprint("graph", __name__)

@graph_bp.route("/grafo")
def visualizar_grafo():
    fuente = request.args.get("fuente", "documents")
    graph_path = Path(f"graphstore/knowledge_graph_{fuente}.gpickle")
    if not graph_path.exists():
        return f"❌ No se encontró el grafo para la fuente '{fuente}'", 404

    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    net = Network(height="700px", width="100%", directed=True, notebook=False)
    net.barnes_hut(gravity=-80000, central_gravity=0.3, spring_length=110, spring_strength=0.10)
    net.toggle_physics(False)  # Desactiva la física para mejorar el rendimiento

    for nodo, data in G.nodes(data=True):
        net.add_node(nodo, label=nodo, title=data.get("tipo", ""))
    for origen, destino, data in G.edges(data=True):
        net.add_edge(origen, destino, title=data.get("fragmento", ""))

    output_file = Path(f"app/static/graph/graph_{fuente}.html")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(output_file))

    return render_template("grafo.html", graph_file=f"graph/graph_{fuente}.html")

@graph_bp.route("/grafo/build", methods=["POST"])
def construir_grafo():
    fuente = request.form.get("fuente", "documents")
    result = subprocess.run([
        "python", "-m", "app.services.build_knowledge_graph", "--source", fuente
    ], capture_output=True, text=True)
    print(result.stdout)
    return redirect(url_for("graph.visualizar_grafo", fuente=fuente))