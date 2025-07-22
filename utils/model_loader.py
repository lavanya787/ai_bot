import os
import json
import gdown

MODELS_DIR = "saved_models"
MODELS_JSON = "models.json"

def list_available_models():
    if not os.path.exists(MODELS_JSON):
        return []
    with open(MODELS_JSON, "r") as f:
        data = json.load(f)
    return list(data.keys())

def load_model_from_drive(model_key):
    with open(MODELS_JSON, "r") as f:
        models = json.load(f)

    model_info = models[model_key]
    model_id = model_info["id"]
    local_path = os.path.join(MODELS_DIR, model_key)

    if os.path.exists(local_path):
        print(f"✅ Model already exists: {local_path}")
        return local_path

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    url = f"https://drive.google.com/uc?id={model_id}"
    print(f"⬇️ Downloading {model_key} from Google Drive...")
    gdown.download(url, local_path, quiet=False)
    return local_path
