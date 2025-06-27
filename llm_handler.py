import torch
import torch.nn as nn
import pickle
import os
import re
import logging
from pythonjsonlogger import jsonlogger
from collections import Counter
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from llm_components.RAGModel import RAGModel
from llm_components.ImprovedBPETokenizer import ImprovedBPETokenizer
from utils.processor import get_chunking_config, chunk_text
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s')
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)

def infer_task_type(prompt: str) -> str:
    prompt_lower = prompt.lower()

    task_keywords = {
        "mcq": ["mcq", "multiple choice", "quiz", "questionnaire", "10 questions", "generate questions", "test questions"],
        "ppt": ["ppt", "slides", "presentation"],
        "summary": ["summarize", "overview", "in brief"],
        "bullet_points": ["bullet", "points", "outline", "key points"],
        "keywords": ["keywords", "terms", "key words"],
        "short_note": ["short note", "note on", "explain shortly"],
        "definition": ["define", "definition", "what is"],
    }

    for task, keywords in task_keywords.items():
        if any(kw in prompt_lower for kw in keywords):
            return task

    return "faq"  # fallback

# Function to count model parameters
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

class LLMHandler:
    def __init__(self, model_path='checkpoint.pt', tokenizer_path='bpe_tokenizer.pkl'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

        self.vocab_size = 5000
        self.d_model = 256
        self.num_heads = 8
        self.num_layers = 8
        self.d_ff = 512
        self.seq_len = 512
        self.dropout = 0.1
        self.pad_token_id = 0
        self.top_k = 3

        self.tokenizer = ImprovedBPETokenizer()
        self.model = RAGModel(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            d_ff=self.d_ff,
            dropout=self.dropout
        ).to(self.device)
        # 👇 Parameter Count
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        total, trainable = count_parameters(self.model)
        logger.info(f"📊 Model Parameters: Total = {total:,}, Trainable = {trainable:,}")
        self.doc_texts = {}
        self.doc_embeddings = {}
        self.is_trained = False
    # Try to load tokenizer safely
        if Path(self.tokenizer_path).exists():
            try:
                self.tokenizer.load(self.tokenizer_path)
                logger.info("✅ Tokenizer loaded successfully.")
            except Exception as e:
                logger.warning(f"⚠️ Tokenizer corrupted: {e}. Deleting and retraining.")
                Path(self.tokenizer_path).unlink(missing_ok=True)
                self.tokenizer = ImprovedBPETokenizer()  # reinitialize
        # Try to load model safely
        if Path(self.model_path).exists():
            try:
                self._load_checkpoint()
            except Exception as e:
                logger.warning(f"⚠️ Model corrupted: {e}. Deleting model.")
                Path(self.model_path).unlink(missing_ok=True)
                self.is_trained = False
        else:
            logger.warning("Model not found. Proceeding without training.")
        if os.path.exists(self.tokenizer_path):
            self.tokenizer.load(self.tokenizer_path)
        else:
            logger.warning("Tokenizer not trained yet. Will train after documents are indexed.")
        if os.path.exists(self.model_path):
            self._load_checkpoint()
        else:
            logger.warning("Model not trained yet. Proceeding untrained.")

    def _load_checkpoint(self):
        try:
            state = torch.load(self.model_path, map_location=self.device)
            if 'model_state_dict' in state:
                self.model.load_state_dict(state['model_state_dict'])
            else:
                self.model.load_state_dict(state)
            self.is_trained = True
            logger.info("✅ Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_trained = False

    def _train_tokenizer(self):
        if not self.doc_texts:
            logger.warning("No documents available to train tokenizer.")
            return
        texts = list(self.doc_texts.values())
        self.tokenizer.train(texts, self.vocab_size)
        self.tokenizer.save(self.tokenizer_path)
        logger.info("Tokenizer trained on uploaded documents.")

    def index_document(self, filename: str, content: str):
        if not content.strip():
            logger.warning(f"Empty document: {filename}")
            return

        logger.info(f"Indexing document: {filename}")

        # Save raw content (for fallback, training)
        self.doc_texts[filename] = {
            "content": content
        }

        # Ensure tokenizer is ready
        if not self.tokenizer.vocab:
            self._train_tokenizer()

        # Get token-level embedding if model is trained
        tokens = self.tokenizer.encode(content)
        if tokens:
            tokens = tokens[:self.seq_len] if len(tokens) > self.seq_len else tokens + [self.pad_token_id] * (self.seq_len - len(tokens))
            tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)
            mask = torch.ones_like(tensor, dtype=torch.float)

            if self.is_trained:
                with torch.no_grad():
                    embedding = self.model.encode_document(tensor, mask)
                    self.doc_embeddings[filename] = embedding.cpu().numpy().flatten()
                    logger.info(f"✅ Token-level embedding stored for: {filename}")

        # -------- Chunking for semantic similarity --------
        config = get_chunking_config(filename)
        chunks = chunk_text(content, chunk_size=config["chunk_size"], mode=config["mode"])
        if not chunks:
            logger.warning(f"No chunks found in: {filename}")
            return

        vectorizer = TfidfVectorizer().fit(chunks)
        embeddings = vectorizer.transform(chunks)

        # Save chunked representation for semantic search
        self.doc_texts[filename].update({
            "sentences": chunks,
            "vectorizer": vectorizer,
            "embeddings": embeddings
        })

        logger.info(f"✅ Indexed {len(chunks)} chunks for semantic search.")

    def train_on_documents(self, epochs=5, batch_size=8, save_path="checkpoint.pt", log_path="training_log.txt"):
        if self.doc_texts:
            logger.warning("No documents available for training.")
            return

        logger.info("Preparing training data...")

        if not self.tokenizer.vocab:
            self._train_tokenizer()

        train_pairs = []
        for text in self.doc_texts.values():
            sentences = re.split(r'[.!?]', text['content'])
            sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
            for i in range(len(sentences) - 1):
                train_pairs.append((sentences[i], sentences[i+1]))

        if not train_pairs:
            logger.warning("No training pairs found.")
            return

        def encode(q, a):
            q_ids = self.tokenizer.encode(q)[:self.seq_len]
            a_ids = self.tokenizer.encode(a)[:self.seq_len]
            q_ids += [self.pad_token_id] * (self.seq_len - len(q_ids))
            a_ids += [self.pad_token_id] * (self.seq_len - len(a_ids))
            return q_ids, a_ids

        encoded = [encode(q, a) for q, a in train_pairs]
        train_data, val_data = train_test_split(encoded, test_size=0.1, random_state=42)

        def to_tensor(data):
            queries = torch.tensor([q for q, _ in data], dtype=torch.long)
            targets = torch.tensor([a for _, a in data], dtype=torch.long)
            return torch.utils.data.TensorDataset(queries, targets)

        train_loader = torch.utils.data.DataLoader(to_tensor(train_data), batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(to_tensor(val_data), batch_size=batch_size)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)

        if os.path.exists(save_path):
            try:
                checkpoint = torch.load(save_path, map_location=self.device)
                if "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    logger.info(f"✅ Resumed training from {save_path}")
                else:
                    self.model.load_state_dict(checkpoint)
            except Exception as e:
                logger.warning(f"⚠️ Failed to load checkpoint: {e}")
        # ✅ Log parameter counts after (re)loading model
        total, trainable = count_parameters(self.model)
        logger.info(f"📊 Model Parameters after load: Total = {total:,}, Trainable = {trainable:,}")
        self.model.train().to(self.device)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as log_file:
            log_file.write("Epoch,TrainLoss,ValLoss\n")

        for epoch in range(epochs):
            total_loss = 0
            for q, a in train_loader:
                q, a = q.to(self.device), a.to(self.device)
                optimizer.zero_grad()
                logits = self.model(q, a)
                loss = loss_fn(logits.view(-1, logits.size(-1)), a.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for q, a in val_loader:
                    q, a = q.to(self.device), a.to(self.device)
                    logits = self.model(q, a)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), a.view(-1))
                    val_loss += loss.item()
                avg_val_loss = val_loss / len(val_loader)
            self.model.train()

            logger.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
            with open(log_path, "a") as log_file:
                log_file.write(f"{epoch+1},{avg_train_loss:.4f},{avg_val_loss:.4f}\n")

        # ✅ Save final model + tokenizer properly
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, save_path)

        tokenizer_path = os.path.splitext(save_path)[0] + "_tokenizer.pkl"
        self.tokenizer.save(tokenizer_path)

        logger.info(f"✅ Model saved to {save_path}")
        logger.info(f"✅ Tokenizer saved to {tokenizer_path}")

        self.is_trained = True

    def extract_chapter_list(self):
        chapter_lines = []
        toc_pattern = re.compile(
            r"(chapter\s+\d+.*?\.+\s*\d+|^\s*\d{1,2}(\.\d+)+\s+.+?\.+\s*\d+)", re.IGNORECASE
        )
        for fname, doc in self.doc_texts.items():
            raw_text = doc.get("content", "")
            lines = raw_text.splitlines()
            for line in lines:
                if toc_pattern.search(line.strip()):
                    cleaned = re.sub(r'\.{2,}', ' ... ', line.strip())
                    chapter_lines.append(cleaned)

        chapter_map = {}

        for line in chapter_lines:
            parts = re.split(r'\s{2,}|\.{2,}', line.strip())
            text = parts[0].strip()
            page = parts[-1].strip() if parts[-1].strip().isdigit() else ""

            match = re.match(r'(\d+(\.\d+)*)(\s+)?(.*)', text)
            if match:
                section = match.group(1)
                title = match.group(4)
                chapter_key = section.split('.')[0]
                if chapter_key not in chapter_map:
                    chapter_map[chapter_key] = []
                chapter_map[chapter_key].append((section, title, page))

        def natural_key(s):
            return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', s)]

        sorted_chapters = sorted(chapter_map.items(), key=lambda x: natural_key(x[0]))
        output = ["📘 Chapters Found:"]

        for chap_num, items in sorted_chapters:
            items = sorted(items, key=lambda x: natural_key(x[0]))
            # Find main chapter title
            main_title = next((t for s, t, p in items if s == chap_num), None)
            if main_title:
                output.append(f"### Chapter {chap_num}: {main_title}")
            else:
                output.append(f"### Chapter {chap_num}")
            # Add sub-sections
            for section, title, page in items:
                if section != chap_num:
                    line = f"  - {section} {title} ... {page}" if page else f"  - {section} {title}"
                    output.append(line)

        return output if output else ["❌ No chapter-like patterns found."]

    def _semantic_snippet_match(self, prompt: str):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        all_sentences = []
        sentence_sources = []

        for fname, doc in self.doc_texts.items():
            text = doc if isinstance(doc, str) else doc.get("content", "")
            clean_text = re.sub(r'\s{2,}', ' ', text)
            sentences = re.split(r'(?<=[.!?])\s+', clean_text)
            for sent in sentences:
                if len(sent.strip()) > 40 and not re.match(r'^\d+(\.\d+)*\s', sent):  # Skip TOC-like lines
                    all_sentences.append(sent.strip())
                    sentence_sources.append(fname)

        if not all_sentences:
            return "❌ No valid content found for semantic analysis."

        vectorizer = TfidfVectorizer().fit(all_sentences + [prompt])
        vectors = vectorizer.transform(all_sentences + [prompt])
        similarities = cosine_similarity(vectors[-1], vectors[:-1]).flatten()

        top_k = 3
        top_indices = similarities.argsort()[-top_k:][::-1]
        top_matches = [(all_sentences[i], similarities[i], sentence_sources[i]) for i in top_indices if similarities[i] > 0.15]

        if not top_matches:
            return "❌ No semantically relevant content found."

        combined_snippet = " ".join([s[0] for s in top_matches])
        related_words = sorted(set(
            w.lower() for s in top_matches for w in re.findall(r'\b[a-zA-Z]{6,}\b', s[0])
        ), key=lambda x: -len(x))[:5]

        avg_conf = np.mean([s[1] for s in top_matches])
        sources = sorted(set(s[2] for s in top_matches))

        return {
            "prompt": prompt,
            "snippet": combined_snippet,
            "confidence": avg_conf,
            "source": ", ".join(sources),
            "topics": related_words
        }
    def llm_summarize_chunks(self, chunks, max_sentences=5):
        if not chunks:
            return "⚠️ No relevant information found to summarize."

        sentences = []
        for ch in chunks:
            for sent in re.split(r'[.!?]', ch):
                sent = sent.strip()
                if len(sent) > 40 and not re.search(r'(chapter|index|includesdesign|www|http|solved problem|exercise)', sent.lower()):
                    sentences.append(sent)

        sentences = sorted(sentences, key=len, reverse=True)[:max_sentences]
        return "\n".join(f"- {s}" for s in sentences) if sentences else "⚠️ No clean content to summarize."


    def _faq_chunk_search(self, prompt: str, top_k: int = 3):
        if not hasattr(self, "semantic_model"):
            from sentence_transformers import SentenceTransformer
            self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

        results = []

        for fname, doc in self.doc_texts.items():
            chunks = doc.get("sentences", [])

            def is_useful(chunk):
                chunk = chunk.strip()
                if len(chunk) < 40:
                    return False
                if re.match(r'^\s*(chapter|table of contents|contents|solved problem|index)', chunk.lower()):
                    return False
                if re.search(r'(\.{5,}|\d{3,})', chunk):
                    return False
                if chunk.count('.') > 10:
                    return False
                digit_ratio = sum(c.isdigit() for c in chunk) / max(len(chunk), 1)
                #if digit_ratio > 0.25:
                 #   return False
                return True

            chunks = [c for c in chunks if is_useful(c)]
            if not chunks:
                continue

            chunk_embeddings = self.semantic_model.encode(chunks)
            query_vec = self.semantic_model.encode([prompt])
            sims = cosine_similarity(query_vec, chunk_embeddings).flatten()
            top_indices = sims.argsort()[-top_k:][::-1]

            for idx in top_indices:
                if sims[idx] > 0.3:  # increased threshold from 0.2 to 0.3
                    results.append({
                        "chunk": chunks[idx],
                        "confidence": sims[idx],
                        "source": fname
                    })

        if not results:
            return {
                "prompt": prompt,
                "summary": "⚠️ No strong semantic match found in the uploaded documents.",
                "confidence": 0.0,
                "source": "N/A",
                "topics": []
            }

        results.sort(key=lambda x: -x["confidence"])
        top_results = results[:top_k]

        top_chunks = [r["chunk"] for r in top_results]
        summary = self.llm_summarize_chunks(top_chunks, max_sentences=5)
        avg_conf = np.mean([r["confidence"] for r in top_results])
        sources = ", ".join(sorted(set(r["source"] for r in top_results)))

        topic_words = sorted(set(
            w.lower() for c in top_chunks for w in re.findall(r'\b[a-zA-Z]{4,}\b', c)
            if w.lower() not in ["includesdesign", "etc", "using", "chapter", "which"]
        ), key=lambda x: -len(x))[:5]

        return {
            "prompt": prompt,
            "summary": summary,
            "confidence": avg_conf,
            "source": sources,
            "topics": topic_words
        }


    # Updated extract_chapter_list for cleaner output
    def extract_chapter_list(self):
        chapter_lines = []
        toc_pattern = re.compile(
            r"(chapter\s+\d+.*?\.+\s*\d+|^\s*\d{1,2}(\.\d+)+\s+.+?\.+\s*\d+)", re.IGNORECASE
        )
        for fname, doc in self.doc_texts.items():
            raw_text = doc.get("content", "")
            lines = raw_text.splitlines()
            for line in lines:
                if toc_pattern.search(line.strip()):
                    cleaned = re.sub(r'\.{2,}', ' ... ', line.strip())
                    chapter_lines.append(cleaned)

        unique_lines = list(dict.fromkeys(chapter_lines))
        return ["📘 Chapters Found:"] + unique_lines if unique_lines else ["❌ No chapter-like patterns found."]

    def generate_response(self, prompt: str, task: str = '') -> str:
        if not self.doc_texts:
            return "📂 Please upload documents before asking questions."
        # Special case: if user explicitly asks for chapters or table of contents
        if any(kw in prompt.lower() for kw in ["list of chapters", "table of contents", "chapter names"]):
            logger.info("📘 Detected request for chapter list.")
            chapters = self.extract_chapter_list()
            return "\n".join(["📘 Chapters Found:"] + chapters)
    
        if not task or task == 'auto':
            task = infer_task_type(prompt)
            logger.info(f"[Auto Task Detection] Prompt: '{prompt}' → Task: '{task}'")

        if task == "mcq":
            return (
                f"### 🧠 Multiple Choice Questions\n"
                f"➡️ Generate 5 multiple choice questions from the topic:\n\n{prompt}"
            )

        elif task == "ppt":
            return (
                f"### 🖼 PowerPoint Slide Points\n"
                f"➡️ Summarize into 5–7 bullet points per slide:\n\n{prompt}"
            )

        elif task == "summary":
            return (
                f"### 📘 Summary\n"
                f"Summarize the topic clearly and concisely:\n\n{prompt}"
            )

        elif task == "bullet_points":
            return (
                f"### 📌 Key Bullet Points\n"
                f"List the main bullet points from the following topic:\n\n{prompt}"
            )

        elif task == "keywords":
            return (
                f"### 🔑 Important Keywords\n"
                f"Extract 10 important keywords related to:\n\n{prompt}"
            )

        elif task == "short_note":
            return (
                f"### ✍️ Short Note\n"
                f"Write a short note (4–5 lines) on the topic:\n\n{prompt}"
            )

        elif task == "definition":
            return (
                f"### 📚 Definition\n"
                f"Define the concept of:\n\n{prompt} in simple terms."
            )

        faq_result = self._faq_chunk_search(prompt)
        if isinstance(faq_result, dict):
            return (
                f"### 📘 Answer\n"
                f"**Prompt:** {prompt}\n\n"
                f"**Summary:**\n{faq_result['summary']}\n\n"
                f"**Confidence:** {faq_result['confidence']*100:.1f}%\n"
                f"**Source(s):** {faq_result['source']}\n"
                f"**Topics:** {', '.join(faq_result['topics']) if faq_result['topics'] else 'N/A'}"
            )
        else:
            return str(faq_result)
                      


    def get_status(self):
        return {
            "trained": self.is_trained,
            "device": str(self.device),
            "documents": len(self.doc_texts),
            "tokenizer_vocab": len(self.tokenizer.vocab),
            "embeddings": len(self.doc_embeddings)
        }