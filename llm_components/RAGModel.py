import os
import fitz  # PyMuPDF
import shutil
import json
import hashlib
from datetime import datetime
from llm_handler import LLMHandler
from utils.domain_detector import detect_domain
from file_processing.processor import extract_text_from_file

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".csv", ".json")
BASE_DIR = "rag_data"
MODELS_DIR = "models"
LOGS_DIR = "logs"


def get_file_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def update_metadata(domain_folder, file_name, file_path):
    metadata_path = os.path.join(domain_folder, "metadata.json")
    metadata = []

    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            print("⚠️ Corrupted metadata. Reinitializing.")

    file_hash = get_file_hash(file_path)
    for entry in metadata:
        if entry.get("hash") == file_hash:
            print("⚠️ Duplicate file detected. Skipping.")
            return False

    metadata.append({
        "file": file_name,
        "hash": file_hash,
        "uploaded_at": datetime.now().isoformat(),
        "source": "user_upload"
    })

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return True

def store_and_train(file_path, base_data_dir=BASE_DIR, models_dir=MODELS_DIR):
    if not os.path.exists(file_path):
        print(f"❌ File does not exist: {file_path}")
        return

    text = extract_text_from_file(file_path)
    if not text.strip():
        print("❌ No extractable content found.")
        return

    domain = detect_domain(text)
    print(f"🌐 Detected domain: {domain}")

    # Folder setup
    domain_folder = os.path.join(base_data_dir, domain)
    os.makedirs(domain_folder, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Save original and text file
    filename = os.path.basename(file_path)
    target_original = os.path.join(domain_folder, filename)
    target_text = os.path.join(domain_folder, f"{os.path.splitext(filename)[0]}.txt")

    # Skip if already processed
    if not update_metadata(domain_folder, filename, file_path):
        return

    # Copy original + text content
    shutil.copy(file_path, target_original)
    with open(target_text, "w", encoding="utf-8") as f:
        f.write(text)

    # Load handler
    handler = LLMHandler()
    handler.model_path = os.path.join(models_dir, f"{domain}_checkpoint.pt")
    handler.tokenizer_path = os.path.join(models_dir, f"{domain}_tokenizer.pkl")

    # Load existing model/tokenizer if available
    if os.path.exists(handler.model_path):
        handler._load_checkpoint()
    if os.path.exists(handler.tokenizer_path):
        handler.tokenizer.load(handler.tokenizer_path)

    # Index all domain documents
    documents = {}
    for fname in os.listdir(domain_folder):
        if fname.endswith(".txt"):
            with open(os.path.join(domain_folder, fname), "r", encoding="utf-8") as f:
                documents[fname] = f.read()

    for fname, content in documents.items():
        handler.index_document(fname, content)

    print("🚀 Training domain model...")
    handler.train_on_documents(
        epochs=5,
        batch_size=8,
        save_path=handler.model_path,
        log_path=os.path.join(LOGS_DIR, f"{domain}_log.txt")
    )
    print(f"✅ Training complete for domain: {domain}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python rag_domain_trainer.py <path_to_file>")
    else:
        store_and_train(sys.argv[1])
