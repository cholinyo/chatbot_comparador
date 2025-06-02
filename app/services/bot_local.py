import os
from llama_cpp import Llama
from app.config.settings import get_local_model_path

_llm = None

def get_local_response(prompt_usuario):
    global _llm
    if _llm is None:
        model_path = get_local_model_path()

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"⚠️ Modelo local no encontrado en: {model_path}")

        _llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=6,
            n_gpu_layers=0,
            verbose=False
        )

    prompt = (
        "<|system|>\nEres un asistente de IA local. No inventes.\n</s>\n"
        f"<|user|>\n{prompt_usuario}</s>\n<|assistant|>\n"
    )
    output = _llm(prompt, max_tokens=512, temperature=0.3, top_k=40, top_p=0.7, stop=["</s>"])
    return output["choices"][0]["text"].strip()
