import os
import pickle
import faiss

def check_vectorstore():
    """Verificar el estado del vectorstore"""
    print("🔍 Diagnosticando RAG...\n")
    
    sources = ["documents", "web", "apis", "bbdd"]
    total_fragments = 0
    
    for source in sources:
        base_path = f"vectorstore/{source}"
        index_path = f"{base_path}/index.faiss"
        fragments_path = f"{base_path}/fragmentos.pkl"
        
        print(f"📁 Fuente: {source}")
        
        # Verificar si existen los archivos
        if os.path.exists(index_path) and os.path.exists(fragments_path):
            try:
                # Cargar índice FAISS
                index = faiss.read_index(index_path)
                print(f"  ✅ Índice FAISS: {index.ntotal} vectores")
                
                # Cargar fragmentos
                with open(fragments_path, "rb") as f:
                    fragments = pickle.load(f)
                print(f"  ✅ Fragmentos: {len(fragments)} textos")
                
                # Mostrar ejemplo
                if fragments:
                    sample = fragments[0][:100] + "..." if len(fragments[0]) > 100 else fragments[0]
                    print(f"  📝 Ejemplo: {sample}")
                
                total_fragments += len(fragments)
                
            except Exception as e:
                print(f"  ❌ Error cargando: {e}")
        else:
            print(f"  ⚠️  No encontrado (necesita ingesta)")
        
        print()
    
    print(f"📊 Total fragmentos disponibles: {total_fragments}")
    
    if total_fragments == 0:
        print("\n🔧 SOLUCIÓN: Necesitas ejecutar los scripts de ingesta:")
        print("  python -m app.services.ingest_documents")
        print("  python -m app.services.ingest_web")
        print("  python -m app.services.ingest_api")
    else:
        print("\n✅ RAG configurado correctamente!")
    
    return total_fragments > 0

def test_rag_search():
    """Probar búsqueda RAG"""
    print("\n🧪 Probando búsqueda RAG...")
    
    try:
        from app.utils.rag_utils import buscar_fragmentos_combinados
        
        # Probar búsqueda
        test_query = "¿Qué información hay disponible?"
        fragments = buscar_fragmentos_combinados(test_query, k=3)
        
        print(f"🔍 Consulta: '{test_query}'")
        print(f"📋 Fragmentos encontrados: {len(fragments)}")
        
        for i, frag in enumerate(fragments[:2]):  # Mostrar solo 2
            print(f"\n  {i+1}. Fuente: {frag.get('fuente', 'unknown')}")
            print(f"     Distancia: {frag.get('distancia', 'N/A'):.3f}")
            print(f"     Texto: {frag.get('texto', '')[:150]}...")
        
        return len(fragments) > 0
        
    except Exception as e:
        print(f"❌ Error en búsqueda RAG: {e}")
        return False

def test_complete_flow():
    """Probar flujo completo incluyendo Ollama"""
    print("\n🚀 Probando flujo completo...")
    
    try:
        from app.routes.chat import chat_bp
        from app.utils.rag_utils import buscar_fragmentos_combinados
        from app.services.bot_local import get_local_response
        
        # Simular flujo del chat
        pregunta = "¿Qué servicios hay disponibles?"
        print(f"❓ Pregunta: {pregunta}")
        
        # 1. Búsqueda RAG
        fragmentos = buscar_fragmentos_combinados(pregunta, k=3)
        print(f"📋 Fragmentos recuperados: {len(fragmentos)}")
        
        # 2. Construir prompt
        contexto = "\n".join([f"- {f['texto']}" for f in fragmentos])
        prompt = f"""Usa la siguiente información para responder a la pregunta:

{contexto}

Pregunta: {pregunta}
Respuesta:"""
        
        print(f"📝 Longitud del prompt: {len(prompt)} caracteres")
        print(f"🧠 Contexto incluido: {'SÍ' if contexto.strip() else 'NO'}")
        
        # 3. Generar respuesta (solo si hay contexto)
        if contexto.strip():
            respuesta = get_local_response(prompt)
            print(f"✅ Respuesta generada: {len(respuesta)} caracteres")
            print(f"🎯 Respuesta: {respuesta[:200]}...")
        else:
            print("⚠️ Sin contexto RAG, respuesta será genérica")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en flujo completo: {e}")
        return False

if __name__ == "__main__":
    # Ejecutar diagnósticos
    has_data = check_vectorstore()
    
    if has_data:
        has_search = test_rag_search()
        if has_search:
            test_complete_flow()
    
    print(f"\n{'='*50}")
    print("🎯 RESUMEN:")
    print("✅ RAG funcional" if has_data else "❌ Necesita ingesta de datos")
    print("💡 Ejecuta 'python test_rag.py' para diagnosticar")