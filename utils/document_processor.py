import os
import re
import pickle
import faiss
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import docx  # python-docx
import speech_recognition as sr
from PIL import Image
import pytesseract
import logging
from typing import List
from concurrent.futures import ThreadPoolExecutor
# Optional: spaCy
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    nlp = None
    SPACY_AVAILABLE = False

logger = logging.getLogger("processor")
logging.basicConfig(level=logging.INFO)
class DocumentProcessor:
    def __init__(self, base_dir="faiss_logs", model_name="all-MiniLM-L6-v2", device="cpu"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.model = SentenceTransformer(model_name)

    def file_type_detector(self, file_name: str) -> str:
        ext = os.path.splitext(file_name)[-1].lower()
        if ext == ".txt":
            return "text"
        elif ext == ".pdf":
            return "pdf"
        elif ext == ".docx":
            return "docx"
        elif ext in [".wav", ".mp3", ".m4a"]:
            return "audio"
        elif ext in [".png", ".jpg", ".jpeg"]:
            return "image"
        else:
            return "unknown"

    def read_file(self, file, file_type: str) -> str:
        if file_type == "text":
            file.seek(0)
            return file.read().decode("utf-8", errors="ignore")
        elif file_type == "pdf":
            file.seek(0)
            text = ""
            with fitz.open(stream=file.read(), filetype="pdf") as doc:
                for page in doc:
                    text += page.get_text()
            return text
        elif file_type == "docx":
            file.seek(0)
            doc = docx.Document(file)
            return "\n".join([para.text for para in doc.paragraphs])
        elif file_type == "audio":
            recognizer = sr.Recognizer()
            with sr.AudioFile(file) as source:
                audio_data = recognizer.record(source)
                try:
                    return recognizer.recognize_google(audio_data)
                except sr.UnknownValueError:
                    return ""
        elif file_type == "image":
            image = Image.open(file)
            return pytesseract.image_to_string(image)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

# Inside class DocumentProcessor:
    
    def chunk_text(self, text: str, chunk_size: int = 1200) -> List[str]:
        if not text or len(text.strip()) < 50:
            return []
    
        try:
            if SPACY_AVAILABLE and nlp:
                doc = nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents]
            else:
                sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        except Exception as e:
            logger.warning(f"Chunk fallback due to error: {e}")
            sentences = text.split(". ")
    
        chunks = []
        current_chunk = ""
    
        def process_chunk(start_idx):
            chunk = ""
            idx = start_idx
            while idx < len(sentences) and len(chunk) + len(sentences[idx]) < chunk_size:
                chunk += sentences[idx] + " "
                idx += 1
            return chunk.strip()
    
        with ThreadPoolExecutor(max_workers=4) as executor:
            i = 0
            futures = []
            while i < len(sentences):
                futures.append(executor.submit(process_chunk, i))
                j = i
                chunk_len = 0
                while j < len(sentences) and chunk_len + len(sentences[j]) < chunk_size:
                    chunk_len += len(sentences[j])
                    j += 1
                i = j
    
            for future in futures:
                chunk = future.result()
                if len(chunk.split()) >= 10:
                    chunks.append(chunk)
    
        return chunks
        
    def build_faiss_index(self, chunks: List[str], file_name: str, batch_size=32) -> str:
        file_id = os.path.splitext(file_name)[0].replace(" ", "_").lower()
        folder = os.path.join(self.base_dir, file_id)

        if os.path.exists(os.path.join(folder, "faiss.index")):
            return folder

        os.makedirs(folder, exist_ok=True)

        embeddings = self.model.encode(
            chunks,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings).astype("float32"))

        faiss.write_index(index, os.path.join(folder, "faiss.index"))
        with open(os.path.join(folder, "chunks.pkl"), "wb") as f:
            pickle.dump(chunks, f)

        return folder

    def process_and_index_file(self, file, file_name: str) -> str:
        file_type = self.file_type_detector(file_name)
        if file_type == "unknown":
            raise ValueError(f"Unsupported file format: {file_name}")

        text = self.read_file(file, file_type)
        chunks = self.process_raw_text_chunks(text)
        if not chunks:
            raise ValueError(f"No valid content extracted from file: {file_name}")
        return self.build_faiss_index(chunks, file_name)

    def search(self, query: str, file_name: str, top_k: int = 5) -> List[str]:
        file_id = os.path.splitext(file_name)[0].replace(" ", "_").lower()
        folder = os.path.join(self.base_dir, file_id)
        if not os.path.exists(folder):
            raise FileNotFoundError(f"No index found for {file_id}")

        index = faiss.read_index(os.path.join(folder, "faiss.index"))
        with open(os.path.join(folder, "chunks.pkl"), "rb") as f:
            chunks = pickle.load(f)

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        D, I = index.search(np.array(query_embedding).astype("float32"), top_k)

        return [chunks[i] for i in I[0]]