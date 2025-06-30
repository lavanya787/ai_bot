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
from sklearn.feature_extraction.text import TfidfVectorizer
from llm_components.RAGModel import RAGModel
from llm_components.ImprovedBPETokenizer import ImprovedBPETokenizer
from utils.processor import get_chunking_config, chunk_text
from models.qa_model import get_embedding  # Use custom embedding

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
    return "faq"

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
        self.seq_len = 512
        self.pad_token_id = 0
        self.top_k = 3

        self.tokenizer = ImprovedBPETokenizer()
        self.model = RAGModel(vocab_size=self.vocab_size).to(self.device)
        
        total, trainable = count_parameters(self.model)
        logger.info(f"📊 Model Parameters: Total = {total:,}, Trainable = {trainable:,}")
        
        self.doc_texts = {}
        self.doc_embeddings = {}
        self.is_trained = False
        
        if Path(self.tokenizer_path).exists():
            try:
                self.tokenizer.load(self.tokenizer_path)
                logger.info("✅ Tokenizer loaded successfully.")
            except Exception as e:
                logger.warning(f"⚠️ Tokenizer corrupted: {e}. Deleting and retraining.")
                Path(self.tokenizer_path).unlink(missing_ok=True)
                self.tokenizer = ImprovedBPETokenizer()
        
        if Path(self.model_path).exists():
            try:
                self._load_checkpoint()
            except Exception as e:
                logger.warning(f"⚠️ Model corrupted: {e}. Deleting model.")
                Path(self.model_path).unlink(missing_ok=True)
                self.is_trained = False
        else:
            logger.warning("Model not found. Proceeding without training.")

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
        self.doc_texts[filename] = {"content": content}
        if not self.tokenizer.vocab:
            self._train_tokenizer()
        tokens = self.tokenizer.encode(content)
        if tokens:
            tokens = tokens[:self.seq_len] if len(tokens) > self.seq_len else tokens + [self.pad_token_id] * (self.seq_len - len(tokens))
            tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)
            if self.is_trained:
                with torch.no_grad():
                    embedding = self.model.encode_document(tensor)
                    self.doc_embeddings[filename] = embedding.cpu().numpy().flatten()
                    logger.info(f"✅ Token-level embedding stored for: {filename}")
        config = get_chunking_config(filename)
        chunks = chunk_text(content, chunk_size=config["chunk_size"], mode=config["mode"])
        if not chunks:
            logger.warning(f"No chunks found in: {filename}")
            return
        vectorizer = TfidfVectorizer().fit(chunks)
        embeddings = vectorizer.transform(chunks)
        self.doc_texts[filename].update({
            "sentences": chunks,
            "vectorizer": vectorizer,
            "embeddings": embeddings
        })
        logger.info(f"✅ Indexed {len(chunks)} chunks for semantic search.")

    def train_on_documents(self, epochs=5, batch_size=8, save_path="checkpoint.pt", log_path="training_log.txt"):
        if not self.doc_texts:
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

    def _semantic_snippet_match(self, prompt: str):
        all_sentences = []
        sentence_sources = []
        for fname, doc in self.doc_texts.items():
            text = doc if isinstance(doc, str) else doc.get("content", "")
            clean_text = re.sub(r'\s{2,}', ' ', text)
            sentences = re.split(r'(?<=[.!?])\s+', clean_text)
            for sent in sentences:
                if len(sent.strip()) > 40 and not re.match(r'^\d+(\.\d+)*\s', sent):
                    all_sentences.append(sent.strip())
                    sentence_sources.append(fname)
        if not all_sentences:
            return "❌ No valid content found for semantic analysis."
        # Use custom embedding instead of SentenceTransformer
        prompt_embedding = get_embedding(prompt)
        sentence_embeddings = [get_embedding(sent) for sent in all_sentences]
        sentence_embeddings = np.array([emb for emb in sentence_embeddings if emb.size > 0])
        if not sentence_embeddings.size:
            return "❌ No valid embeddings generated."
        similarities = cosine_similarity([prompt_embedding], sentence_embeddings).flatten()
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
        all_chunks = []
        chunk_sources = []
        for fname, doc in self.doc_texts.items():
            chunks = doc.get("sentences", [])
            chunks = [c for c in chunks if (
                len(c.strip()) >= 40 and
                not re.match(r'^\s*(chapter|table of contents|contents|solved problem|index)', c.lower()) and
                not re.search(r'(\.{5,}|\d{3,})', c) and
                c.count('.') <= 10
            )]
            all_chunks.extend(chunks)
            chunk_sources.extend([fname] * len(chunks))
        if not all_chunks:
            return {
                "prompt": prompt,
                "summary": "⚠️ No strong semantic match found in the uploaded documents.",
                "confidence": 0.0,
                "source": "N/A",
                "topics": []
            }
        # Use custom embedding
        prompt_embedding = get_embedding(prompt)
        chunk_embeddings = [get_embedding(chunk) for chunk in all_chunks]
        chunk_embeddings = np.array([emb for emb in chunk_embeddings if emb.size > 0])
        if not chunk_embeddings.size:
            return {
                "prompt": prompt,
                "summary": "⚠️ No valid embeddings generated.",
                "confidence": 0.0,
                "source": "N/A",
                "topics": []
            }
        sims = cosine_similarity([prompt_embedding], chunk_embeddings).flatten()
        top_indices = sims.argsort()[-top_k:][::-1]
        top_results = [
            {"chunk": all_chunks[i], "confidence": sims[i], "source": chunk_sources[i]}
            for i in top_indices if sims[i] > 0.3
        ]
        if not top_results:
            return {
                "prompt": prompt,
                "summary": "⚠️ No strong semantic match found in the uploaded documents.",
                "confidence": 0.0,
                "source": "N/A",
                "topics": []
            }
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
        if any(kw in prompt.lower() for kw in ["list of chapters", "table of contents", "chapter names"]):
            logger.info("📘 Detected request for chapter list.")
            chapters = self.extract_chapter_list()
            return "\n".join(["📘 Chapters Found:"] + chapters)
        if not task or task == 'auto':
            task = infer_task_type(prompt)
            logger.info(f"[Auto Task Detection] Prompt: '{prompt}' → Task: '{task}'")
        if task == "mcq":
            return f"### 🧠 Multiple Choice Questions\n➡️ Generate 5 multiple choice questions from the topic:\n\n{prompt}"
        elif task == "ppt":
            return f"### 🖼 PowerPoint Slide Points\n➡️ Summarize into 5–7 bullet points per slide:\n\n{prompt}"
        elif task == "summary":
            return f"### 📘 Summary\nSummarize the topic clearly and concisely:\n\n{prompt}"
        elif task == "bullet_points":
            return f"### 📌 Key Bullet Points\nList the main bullet points from the following topic:\n\n{prompt}"
        elif task == "keywords":
            return f"### 🔑 Important Keywords\nExtract 10 important keywords related to:\n\n{prompt}"
        elif task == "short_note":
            return f"### ✍️ Short Note\nWrite a short note (4–5 lines) on the topic:\n\n{prompt}"
        elif task == "definition":
            return f"### 📚 Definition\nDefine the concept of:\n\n{prompt} in simple terms."
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