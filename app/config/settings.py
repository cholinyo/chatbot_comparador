import os
import json

SETTINGS_PATH = os.path.join("app", "config", "settings.json")

def load_settings():
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_settings(data):
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def get_openai_model():
    return load_settings().get("modelo_openai", "gpt-4")

def get_local_model_path():
    modelo_relativo = load_settings().get("modelo_local")
    return os.path.join("models", modelo_relativo)
