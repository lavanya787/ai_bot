import os
import json
import csv
import pickle
import logging
from datetime import datetime
from pathlib import Path

import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from transformers import Trainer

# Safe import for Streamlit
try:
    import streamlit as st
    USE_STREAMLIT = True
except ImportError:
    USE_STREAMLIT = False

# ----------------------------
# Text Generation Evaluation
# ----------------------------
def evaluate_generation(predictions, references):
    smooth_fn = SmoothingFunction().method1
    bleu_scores = [
        sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth_fn)
        for pred, ref in zip(predictions, references)
    ]
    rouge = Rouge()
    rouge_scores = rouge.get_scores(predictions, references, avg=True)
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

    return {
        "BLEU": round(avg_bleu, 4),
        "ROUGE-1": round(rouge_scores["rouge-1"]["f"], 4),
        "ROUGE-2": round(rouge_scores["rouge-2"]["f"], 4),
        "ROUGE-L": round(rouge_scores["rouge-l"]["f"], 4),
    }

# ----------------------------
# Logging Utilities
# ----------------------------
def log_to_json(path, new_entry):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logs = []
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            logs = []
    logs.append(new_entry)
    with open(path, "w") as f:
        json.dump(logs, f, indent=4)

def log_to_csv(path, new_entry, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(new_entry)

def evaluate_and_log(predictions, references, model_name="default_model", log_dir="logs"):
    metrics = evaluate_generation(predictions, references)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = {"timestamp": timestamp, "model": model_name, **metrics}
    log_to_json(os.path.join(log_dir, "evaluation_logs.json"), record)
    log_to_csv(os.path.join(log_dir, "evaluation_logs.csv"), record, fieldnames=record.keys())
    return record

# ----------------------------
# Dataset for Transformers
# ----------------------------
class EvalDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# ----------------------------
# Confusion Matrix Plot
# ----------------------------


# ----------------------------
# Transformer Classifier Eval
# ----------------------------
def evaluate_transformer(model, tokenizer, eval_df):
    label_mapping = {label: idx for idx, label in enumerate(sorted(eval_df["label"].unique()))}
    labels = eval_df["label"].map(label_mapping).tolist()
    encodings = tokenizer(eval_df["text"].tolist(), truncation=True, padding=True, max_length=128)
    eval_dataset = EvalDataset(encodings, labels)

    trainer = Trainer(model=model)
    raw_preds = trainer.predict(eval_dataset)
    preds = raw_preds.predictions.argmax(-1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    report = classification_report(labels, preds, target_names=list(label_mapping.keys()), output_dict=True)

    return {
        "accuracy": acc,
        "f1_score": f1,
        "classification_report": report
    }

# ----------------------------
# LSTM Evaluation
# ----------------------------
def evaluate_lstm_classifier(model, dataloader, label_names=None):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in dataloader:
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    report = classification_report(all_labels, all_preds, target_names=label_names or [], output_dict=True)


    if USE_STREAMLIT:
        st.write(f"🔍 **Accuracy:** {acc:.4f} | **F1 Score:** {f1:.4f}")
    return {
        "accuracy": acc,
        "f1_score": f1,
        "classification_report": report
    }

# ----------------------------
# Unified Evaluation Handler
# ----------------------------
def smart_evaluate(task_type, **kwargs):
    if task_type == "classification":
        return evaluate_transformer(
            model=kwargs["model"],
            tokenizer=kwargs["tokenizer"],
            eval_df=kwargs["eval_df"]
        )
    elif task_type == "lstm_classification":
        return evaluate_lstm_classifier(
            model=kwargs["model"],
            dataloader=kwargs["dataloader"],
            label_names=kwargs.get("label_names")
        )
    elif task_type == "generation":
        return evaluate_and_log(
            predictions=kwargs["predictions"],
            references=kwargs["references"],
            model_name=kwargs.get("model_name", "default_model"),
            log_dir=kwargs.get("log_dir", "logs")
        )
    else:
        raise ValueError(f"❌ Unknown task_type: {task_type}")

# ----------------------------
# Save .pkl to Domain Folder
# ----------------------------
def save_checkpoint(model, domain: str, filename: str):
    """
    Saves a model as rag_data/{domain}/{domain}_{filename}.pkl
    """
    folder = Path(f"rag_data/{domain}")
    folder.mkdir(parents=True, exist_ok=True)
    
    # Ensure filename doesn't include extension
    base_name = Path(filename).stem
    full_name = f"{domain}_{base_name}.pkl"
    
    model_path = folder / full_name
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    logging.info(f"✅ Model saved to {model_path}")
    return str(model_path)


# ----------------------------
# Log Metrics to CSV/JSON
# ----------------------------
def log_classification_metrics(metrics: dict, model_name: str, domain: str):
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "domain": domain,
        "accuracy": round(metrics["accuracy"], 4),
        "f1_score": round(metrics["f1_score"], 4),
    }
    json_path = f"rag_data/{domain}/classification_eval.json"
    csv_path = f"rag_data/{domain}/classification_eval.csv"
    log_to_json(json_path, record)
    log_to_csv(csv_path, record, fieldnames=record.keys())
    logging.info(f"📁 Evaluation metrics logged to {csv_path}")
