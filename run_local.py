import os
from llama_cpp import Llama

MODEL_PATH = os.path.join("models", "llama3-8b", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")

_llm = None

def get_local_response(prompt_usuario):
    global _llm
    if _llm is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modelo local no encontrado en: {MODEL_PATH}")
        _llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=6, n_gpu_layers=0, verbose=False)

    system_prompt = (
        "Eres un asistente de IA desplegado localmente. "
        "No tienes acceso a Internet. Solo respondes con información verificada. "
        "Si no sabes algo, responde 'No tengo suficiente información'."
    )

    prompt = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{prompt_usuario}</s>\n<|assistant|>\n"
    output = _llm(prompt, max_tokens=512, temperature=0.3, top_k=40, top_p=0.7, stop=["</s>"])
    return output["choices"][0]["text"].strip()
if __name__ == "__main__":
    pregunta = "¿Cuál es la capital de Francia?"
    respuesta = get_local_response(pregunta)
    print("\n🧠 Respuesta del modelo local:\n")
    print(respuesta)