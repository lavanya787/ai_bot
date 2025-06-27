import os
import fitz  # PyMuPDF
import shutil
import json
import hashlib
from datetime import datetime
from llm_handler import LLMHandler
from utils.domain_detector import detect_domain  # Your existing function


def extract_text_from_pdf(file_path):
    if file_path.endswith(".txt"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            print(f"❌ Failed to read TXT file '{file_path}': {e}")
            return ""

    try:
        doc = fitz.open(file_path)
        text = "\n".join(page.get_text() for page in doc)
        return text.strip()
    except Exception as e:
        print(f"❌ Failed to open PDF '{file_path}': {e}")
        return ""



def get_file_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def update_metadata(domain_folder, file_name, file_path):
    metadata_path = os.path.join(domain_folder, "metadata.json")
    metadata = []

    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    file_hash = get_file_hash(file_path)
    for entry in metadata:
        if entry["hash"] == file_hash:
            print("⚠️ Duplicate file detected. Skipping retraining.")
            return False

    metadata.append({
        "file": file_name,
        "uploaded_at": datetime.now().isoformat(),
        "source": "user_upload",
        "hash": file_hash
    })

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return True


def store_and_train(file_path, base_data_dir="rag_data", models_dir="models"):
    handler = LLMHandler()
    text = extract_text_from_pdf(file_path)
    if not text:
        print("❌ Could not extract text from PDF.")
        return

    domain = detect_domain(text)
    domain_folder = os.path.join(base_data_dir, domain)
    os.makedirs(domain_folder, exist_ok=True)

    filename = os.path.basename(file_path)
    new_pdf_path = os.path.join(domain_folder, filename)
    shutil.copy(file_path, new_pdf_path)

    text_file_path = os.path.join(domain_folder, filename.replace('.pdf', '.txt'))
    with open(text_file_path, "w", encoding="utf-8") as f:
        f.write(text)

    if not update_metadata(domain_folder, filename, file_path):
        return

    model_path = os.path.join(models_dir, f"{domain}_checkpoint.pt")
    tokenizer_path = os.path.join(models_dir, f"{domain}_tokenizer.pkl")

    handler.model_path = model_path
    handler.tokenizer_path = tokenizer_path

    if os.path.exists(model_path):
        handler._load_checkpoint()
    if os.path.exists(tokenizer_path):
        handler.tokenizer.load(tokenizer_path)

    documents = {}
    for fname in os.listdir(domain_folder):
        if fname.endswith(".txt"):
            fpath = os.path.join(domain_folder, fname)
            with open(fpath, encoding="utf-8") as f:
                documents[fname] = f.read()

    for fname, content in documents.items():
        handler.index_document(fname, content)

    print("🚀 Training (or continuing) domain model...")
    handler.train_on_documents(
        epochs=5,
        batch_size=8,
        save_path=model_path,
        log_path=os.path.join("logs", f"{domain}_log.txt")
    )
    print(f"✅ Finished training for domain '{domain}'.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python rag_domain_trainer.py <path_to_pdf>")
    else:
        store_and_train(sys.argv[1])
