import os
import json
import joblib
import pandas as pd
from datetime import datetime

def detect_domain(text):
    text = text.lower()
    if any(word in text for word in ["momentum", "velocity", "gravity", "compton", "quantum", "wave", "optics"]):
        return "Physics"
    elif any(word in text for word in ["atom", "molecule", "reaction", "acid", "base", "compound", "organic"]):
        return "Chemistry"
    elif any(word in text for word in ["equation", "algebra", "geometry", "calculus", "derivative", "integral"]):
        return "Mathematics"
    elif any(word in text for word in ["cell", "organism", "dna", "protein", "evolution", "mitosis"]):
        return "Biology"
    elif any(word in text for word in ["solar", "ecosystem", "force", "matter", "weather"]):
        return "Science"
    else:
        return "General"

def save_model_and_metrics(model, metrics, domain, model_name, raw_df=None):
    base_dir = os.path.join("trained_models", "Education", domain)
    os.makedirs(base_dir, exist_ok=True)

    # Save model file
    model_path = os.path.join(base_dir, f"{model_name}.pkl")
    joblib.dump(model, model_path)

    # Save metrics JSON
    metrics_path = os.path.join(base_dir, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Save original data
    if raw_df is not None:
        raw_df.to_csv(os.path.join(base_dir, f"{model_name}_data.csv"), index=False)

    # Summary text
    with open(os.path.join(base_dir, f"{model_name}_summary.txt"), "w") as f:
        f.write(f"Model: {model_name}\ndomain: {domain}\nAccuracy: {metrics.get('accuracy')}\nF1 Score: {metrics.get('f1_score')}")

def log_training_summary(model_name, domain, metrics):
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/education_{domain.lower()}_training_log.csv"
    row = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "domain": domain,
        **metrics
    }
    df = pd.DataFrame([row])
    if os.path.exists(log_file):
        df.to_csv(log_file, mode="a", header=False, index=False)
    else:
        df.to_csv(log_file, index=False)

def log_query(query, domain, response, intent):
    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/education_{domain.lower()}_query_log.csv"
    row = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "domain": domain,
        "intent": intent,
        "response": response
    }
    df = pd.DataFrame([row])
    if os.path.exists(log_path):
        df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df.to_csv(log_path, index=False)