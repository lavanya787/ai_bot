import torch
import torch.nn as nn
from datetime import datetime
import os
import re
import logging
import json
import time
import uuid
import shutil
import psutil
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import hashlib
import pickle
import tempfile
import streamlit as st
from tokenizer.uml_tokenizer import UnigramTokenizer
from llm_components.RAGModel import RAGModel
from memory.memory import Memory

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Helper functions
def format_memory_stats():
    try:
        memory = psutil.virtual_memory()
        return f"Memory: {memory.percent}% used ({memory.used/1024/1024/1024:.1f}GB/{memory.total/1024/1024/1024:.1f}GB)"
    except:
        return "Memory stats unavailable"

def log_gpu_stats(logger):
    if torch.cuda.is_available():
        logger.info(f"GPU Memory: {torch.cuda.memory_allocated()/1024/1024/1024:.1f}GB allocated")
    else:
        logger.info("No GPU available")

def get_embedding(text: str):
    try:
        vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        corpus = [text, "dummy document"]
        embeddings = vectorizer.fit_transform(corpus)
        return embeddings[0].toarray().flatten()
    except:
        words = text.lower().split()
        return np.array([len(words), len(set(words)), len(text)])

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def _content_hash(text: str):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def infer_task_type(prompt: str) -> str:
    prompt_lower = prompt.lower()
    task_keywords = {
        "definition": ["define", "definition", "what is", "state", "law", "principle", "rule", "theorem"],
        "mcq": ["mcq", "multiple choice", "quiz", "questionnaire", "generate questions", "test questions"],
        "ppt": ["ppt", "slides", "presentation"],
        "summary": ["summarize", "overview", "in brief"],
        "bullet_points": ["bullet", "points", "outline", "key points"],
        "keywords": ["keywords", "terms", "key words"],
        "short_note": ["short note", "note on", "explain shortly"],
        "explain": ["explain", "describe", "how does", "what happens"],
        "formula": ["formula", "equation", "expression"],
        "law": ["law", "principle", "rule", "theorem"],
        "exercises": ["exercise", "practice", "problems", "questions"],
        "explanation": ["explain", "describe", "how does", "what happens"],
        "answer": ["answer", "respond", "reply", "solution", "resolve", "clarify"],
        "numerical": ["calculate", "find", "what is the", "how much", "velocity", "acceleration", "speed", "distance", "time"],
        "creative": ["story", "lesson plan", "scavenger hunt", "teaching script", "analogies", "analogy"],
        "translate": ["translate", "in tamil", "in spanish", "language"]
    }
    for task, keywords in task_keywords.items():
        if any(kw in prompt_lower for kw in keywords):
            return task
    return "general"

class LLMHandler:
    def __init__(self, model_path='checkpoint.pt', tokenizer_path='uml_tokenizer.pkl', model_version=None):
        logger.info("Initializing LLMHandler")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Device: {self.device}")
        
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.models_trained = "models_trained.json"
        self.model_version = model_version
    
        self.vocab_size = 10000
        self.seq_len = 512
        self.pad_token_id = 0
        self.sos_token_id = 2
        self.eos_token_id = 3
        self.top_k = 5
    
        self.tokenizer = UnigramTokenizer()
        self.model = RAGModel(vocab_size=self.vocab_size).to(self.device)
        self.model.tokenizer = self.tokenizer
    
        self.doc_texts = {}
        self.doc_embeddings = {}
        self.is_trained = False
        self.memory = Memory()
        self.session_id = str(uuid.uuid4())
    
        self._log_model_params()
        self._load_model(model_version)
        self._load_tokenizer()
        
    def _log_model_params(self):
        total, trainable = count_parameters(self.model)
        logger.info(f"📊 Model Parameters: Total = {total:,}, Trainable = {trainable:,}")

    def _load_model(self, domain_name=None):
        if domain_name is None:
            logger.warning("No domain name provided. Using default model.")
            path = self.model_path
        else:
            path = f"saved_models/{domain_name}/{domain_name}_checkpoint.pt"

        if not os.path.exists(path):
            logger.info(f"Model file {path} does not exist, will train new model")
            return False

        try:
            state = torch.load(path, map_location=self.device)
            state_dict = state.get("model_state_dict", state)
            self.model.load_state_dict(state_dict, strict=False)
            self.doc_texts = state.get("doc_texts", {})
            self.is_trained = state.get("is_trained", False)
            if "vocab_size" in state:
                self.vocab_size = state["vocab_size"]
            logger.info(f"✅ Successfully loaded model from {path}")
            logger.info(f"📄 Loaded {len(self.doc_texts)} documents from checkpoint")
            return True
        except Exception as e:
            logger.warning(f"❌ Failed to load model from {path}: {e}")
            logger.info("Reinitializing model due to loading failure")
            self.model = RAGModel(vocab_size=self.vocab_size).to(self.device)
            self.model.tokenizer = self.tokenizer
            return False

    def _load_tokenizer(self, tokenizer_version=None):
        root_tokenizer_path = Path("uml_tokenizer.pkl")
        if root_tokenizer_path.exists():
            try:
                self.tokenizer.load_pickle(str(root_tokenizer_path))
                self.model.tokenizer = self.tokenizer
                logger.info(f"✅ Tokenizer loaded from {root_tokenizer_path}")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to load tokenizer: {e}", exc_info=True)
                raise RuntimeError("Tokenizer loading failed.")
        else:
            logger.error("❌ Tokenizer file 'uml_tokenizer.pkl' not found.")
            raise FileNotFoundError("Tokenizer file 'uml_tokenizer.pkl' is required.")

    def _train_tokenizer(self):
        if not self.doc_texts:
            logger.error("No documents available for tokenizer training")
            return False
        corpus = []
        for fname, doc in self.doc_texts.items():
            content = doc.get("content", "")
            if content.strip():
                corpus.append(content)
        if not corpus:
            logger.error("No valid content for tokenizer training")
            return False
        try:
            self.tokenizer.train(corpus, vocab_size=self.vocab_size)
            logger.info(f"✅ Tokenizer trained with vocab size: {len(self.tokenizer.token_to_id)}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to train tokenizer: {e}")
            return False

    def index_documents(self, documents, force_reindex=False):
        if isinstance(documents, tuple):
            documents = [documents]
        if not documents:
            logger.error("No documents provided for indexing")
            return False
        indexed = 0
        for filename, content in documents:
            if not isinstance(content, str) or not content.strip():
                logger.error(f"Invalid content for {filename}: type={type(content)}")
                continue
            content_hash = _content_hash(content)
            if not force_reindex and filename in self.doc_texts and self.doc_texts[filename].get("hash") == content_hash:
                logger.info(f"Skipping duplicate document: {filename}")
                continue
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            if not sentences:
                logger.warning(f"No valid sentences in {filename}")
                continue
            self.doc_texts[filename] = {
                "content": content,
                "sentences": sentences,
                "hash": content_hash
            }
            logger.info(f"Indexed document: {filename}")
            indexed += 1
        if indexed > 0 and not self.tokenizer.trained:
            self._train_tokenizer()
        logger.info(f"Indexing complete: {indexed} documents indexed")
        return indexed > 0
    
    def train_on_documents(self, epochs=5, batch_size=8, save_path=None, patience=5, domain_name=None):
        logger.info(f"Starting training with {len(self.doc_texts)} documents")
        logger.info(f"Tokenizer status: trained={self.tokenizer.trained}, vocab_size={len(self.tokenizer.token_to_id)}")
        for fname, doc in self.doc_texts.items():
            logger.info(f"Document {fname}: {doc['content'][:100]}...")
        if not self.doc_texts:
            logger.error("No documents available for training")
            st.error("Training failed: No documents available.")
            return False

        if domain_name is None:
            first_filename = next(iter(self.doc_texts), "default")
            base_name = os.path.splitext(first_filename)[0]
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
            domain_name = f"model_{base_name}_{timestamp}"

        save_dir = os.path.join("saved_models", domain_name)
        os.makedirs(save_dir, exist_ok=True)
        if not os.access(save_dir, os.W_OK):
            logger.error(f"No write permission for {save_dir}")
            st.error(f"Training failed: No write permission for {save_dir}")
            return False
        self.model_path = os.path.join(save_dir, f"{domain_name}_checkpoint.pt")
        #self.tokenizer_path = os.path.join(save_dir, "uml_tokenizer.pkl")  # Avoid overwriting default tokenizer path
        if save_path is None:
            save_path = self.model_path
        logger.info(f"Saving model to {self.model_path}")
        logger.info(f"Saving tokenizer to {self.tokenizer_path}")

        # Ensure tokenizer is trained
        if not self.tokenizer.trained or len(self.tokenizer.token_to_id) <= 4:
            logger.info("Tokenizer not trained or vocabulary too small, training tokenizer")
            if not self._train_tokenizer():
                logger.error("Tokenizer training failed, cannot proceed with model training")
                st.error("Training failed: Tokenizer training failed.")
                return False

        train_pairs = []
        for fname, doc in self.doc_texts.items():
            content = doc.get("content", "")
            if not content.strip():
                logger.warning(f"Skipping empty content in {fname}")
                continue
            try:
                token_ids = self.tokenizer.encode_ids(content)
                if not token_ids or len(token_ids) < 10:
                    logger.warning(f"No or insufficient tokens generated for {fname}: {len(token_ids)} tokens")
                    continue
                if not all(isinstance(i, int) and 0 <= i < self.vocab_size for i in token_ids):
                    logger.error(f"Invalid token IDs in {fname}: {token_ids[:10]}...")
                    continue
                if len(token_ids) < self.seq_len * 2:
                    logger.warning(f"Document {fname} too short for training: {len(token_ids)} tokens")
                    continue
                for i in range(0, len(token_ids) - self.seq_len * 2, self.seq_len):
                    input_chunk = token_ids[i: i + self.seq_len]
                    output_chunk = token_ids[i + self.seq_len: i + self.seq_len * 2]
                    input_chunk += [self.pad_token_id] * (self.seq_len - len(input_chunk))
                    output_chunk += [self.pad_token_id] * (self.seq_len - len(output_chunk))
                    train_pairs.append((input_chunk, output_chunk))
            except Exception as e:
                logger.error(f"Error tokenizing {fname}: {e}", exc_info=True)
        if not train_pairs:
            logger.error("No valid training pairs found")
            st.error("Training failed: No valid training pairs found.")
            return False
        logger.info(f"Prepared {len(train_pairs)} training chunks")

        # Adjust batch size for small datasets
        batch_size = min(batch_size, len(train_pairs))
        if batch_size < 1:
            batch_size = 1
        # Adjust epochs for small datasets
        if len(train_pairs) < 10:
            epochs = min(epochs, 20)
            logger.info(f"Small dataset detected, reducing epochs to {epochs}")

        train_data, val_data = train_test_split(train_pairs, test_size=0.1, random_state=42)
        if not val_data:
            logger.warning("Validation set is empty, creating minimal validation set")
            train_data = train_pairs[:-1] if len(train_pairs) > 1 else train_pairs
            val_data = train_pairs[-1:] if train_pairs else []

        def create_dataset(data):
            queries = torch.tensor([q for q, _ in data], dtype=torch.long)
            answers = torch.tensor([a for _, a in data], dtype=torch.long)
            return torch.utils.data.TensorDataset(queries, answers)

        train_loader = torch.utils.data.DataLoader(create_dataset(train_data), batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(create_dataset(val_data), batch_size=batch_size)
        if len(train_loader) == 0 or len(val_loader) == 0:
            logger.error(f"Empty data loader: train_loader={len(train_loader)}, val_loader={len(val_loader)}")
            st.error("Training failed: Empty data loader.")
            return False

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)

        best_val_loss = float('inf')
        no_improvement = 0
        self.model.train()
        st.write("Training started...")

        for epoch in range(epochs):
            total_train_loss = 0.0
            for batch_idx, (queries, answers) in enumerate(train_loader):
                queries, answers = queries.to(self.device), answers.to(self.device)
                optimizer.zero_grad()
                try:
                    logits = self.model(queries, answers)
                    loss = criterion(logits.reshape(-1, logits.size(-1)), answers.reshape(-1))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_train_loss += loss.item()
                except Exception as e:
                    logger.error(f"Error in training batch {batch_idx}: {e}")
                    continue
            avg_train_loss = total_train_loss / max(len(train_loader), 1)

            self.model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for queries, answers in val_loader:
                    queries, answers = queries.to(self.device), answers.to(self.device)
                    try:
                        logits = self.model(queries, answers)
                        val_loss = criterion(logits.view(-1, logits.size(-1)), answers.view(-1))
                        total_val_loss += val_loss.item()
                    except Exception as e:
                        logger.error(f"Error in validation batch: {e}")
                        continue
            avg_val_loss = total_val_loss / max(len(val_loader), 1)

            logger.info(f"[Epoch {epoch+1}] ✅ Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            st.write(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improvement = 0
                try:
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'vocab_size': self.vocab_size,
                        'doc_texts': self.doc_texts,
                        'is_trained': True
                    }, save_path)
                    logger.info(f"✅ New best model saved to {save_path}")
                    st.success(f"New best model saved to {save_path}")
                except Exception as e:
                    logger.error(f"❌ Error saving checkpoint: {e}", exc_info=True)
                    st.error(f"Training failed: Error saving checkpoint - {e}")
                    return False
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    st.warning(f"Early stopping at epoch {epoch+1}")
                    break

        self.is_trained = True
        logger.info("✅ Training complete")
        st.success("Training complete!")
        return True

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
    
    def _search_documents(self, query):
        if not self.doc_texts:
            logger.warning("No documents available for search")
            return None
            
        all_chunks = []
        chunk_sources = []
        chunk_sentences = []
        query_lower = query.lower()
        key_terms = [term.strip() for term in query_lower.split() if len(term) > 3]
        
        for fname, doc in self.doc_texts.items():
            sentences = doc.get("sentences", [])
            content = doc.get("content", "")
            current_chunk = ""
            current_sentences = []
            for sentence in sentences:
                if len(sentence.strip()) < 20:
                    continue
                cleaned = re.sub(r'\(cid\d+\)', '', sentence)
                cleaned = re.sub(r'[^\w\s,.!?()-]', ' ', cleaned)
                cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                if cleaned and len(cleaned) > 20:
                    if (any(term in cleaned.lower() for term in key_terms) or 
                        re.search(r'\b[A-Za-z]+(\'?s)?\s+law\b', cleaned.lower()) or
                        any(phrase in cleaned.lower() for phrase in ["is defined as", "states that", "principle", "refers to", "equation"])):
                        if len(current_chunk) + len(cleaned) < 500:
                            current_chunk += cleaned + ". "
                            current_sentences.append(cleaned)
                        else:
                            all_chunks.append(current_chunk.strip())
                            chunk_sources.append(fname)
                            chunk_sentences.append(current_sentences)
                            current_chunk = cleaned + ". "
                            current_sentences = [cleaned]
            if current_chunk:
                all_chunks.append(current_chunk.strip())
                chunk_sources.append(fname)
                chunk_sentences.append(current_sentences)
        
        if not all_chunks:
            logger.warning("No relevant chunks found")
            return None
        
        query_embedding = get_embedding(query)
        similarities = []
        term_weights = {term: 1.5 for term in key_terms}
        
        for chunk in all_chunks:
            chunk_embedding = get_embedding(chunk)
            try:
                if len(query_embedding) != len(chunk_embedding):
                    max_len = max(len(query_embedding), len(chunk_embedding))
                    query_embedding = np.pad(query_embedding, (0, max_len - len(query_embedding)))
                    chunk_embedding = np.pad(chunk_embedding, (0, max_len - len(chunk_embedding)))
                sim = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
                for term in term_weights:
                    if term in chunk.lower():
                        sim *= term_weights[term]
                if any(phrase in chunk.lower() for phrase in ["is defined as", "states that", "law"]):
                    sim *= 1.3
                similarities.append(sim)
            except:
                similarities.append(0.0)
        
        top_indices = np.argsort(similarities)[-self.top_k:][::-1]
        top_chunks = []
        top_sentences = []
        top_similarities = []
        for i in top_indices:
            if similarities[i] > 0.8:
                top_chunks.append(all_chunks[i])
                top_sentences.extend(chunk_sentences[i])
                top_similarities.append(similarities[i])
        
        if not top_chunks:
            logger.warning("No chunks above similarity threshold")
            return None
        
        combined_text = " ".join(top_chunks)
        avg_confidence = np.mean([similarities[i] for i in top_indices if similarities[i] > 0.5]) if any(s > 0.5 for s in similarities) else 0.0
        sources = list(set(chunk_sources[i] for i in top_indices if similarities[i] > 0.5))
        
        return {
            "content": combined_text[:1000] + "..." if len(combined_text) > 1000 else combined_text,
            "confidence": avg_confidence,
            "sources": sources,
            "sentences": top_sentences[:5]
        }
    
    def generate_response(self, prompt, task="answer", max_length=10000, temperature=0.8, context=None):
        logger.info(f"Generating response for prompt: {prompt[:50]}...")
        logger.info(f"Documents available: {len(self.doc_texts)}")
        logger.info(f"Model trained: {self.is_trained}")
        logger.info(f"Tokenizer trained: {self.tokenizer.trained}")

        if not self.doc_texts:
            logger.warning("No documents available for response generation")
            return "Please upload documents first to enable response generation."

        inferred_task = infer_task_type(prompt)
        logger.info(f"Inferred task type: {inferred_task}")

        context_result = self._search_documents(prompt)
        if not context_result:
            context_result = {
                "content": "No relevant information found in the provided documents.",
                "sources": ["unknown"],
                "confidence": 0.0,
                "sentences": []
            }
            logger.warning("No relevant information found in documents")
        
        context = []
        if Memory and self.memory:
            context = self.memory.get_context(self.session_id, limit=5)

        if self.is_trained and self.tokenizer.trained:
            try:
                prompt_ids = self.tokenizer.encode_ids(prompt)
                if not prompt_ids or not all(isinstance(i, int) for i in prompt_ids):
                    logger.warning("Invalid prompt encoding, falling back to retrieval")
                    return self._format_professional_response(context_result, inferred_task, prompt)

                prompt_ids = prompt_ids[:self.seq_len]
                prompt_ids += [self.pad_token_id] * (self.seq_len - len(prompt_ids))
                prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_ids=prompt_tensor,
                        max_length=max_length,
                        temperature=temperature,
                        top_k=50
                    )

                if generated_ids.numel() > 0:
                    generated_text = self.tokenizer.decode(generated_ids[0].tolist())
                    generated_text = self._post_process_response(generated_text)
                    if len(generated_text.strip()) > 50:
                        context_result["content"] = generated_text
                        context_result["sources"] = context_result.get("sources", ["model"])
                        context_result["confidence"] = context_result.get("confidence", 0.9)
                        response = self._format_professional_response(context_result, inferred_task, prompt)
                        logger.info(f"Generated response length: {len(response)}")

                        if Memory and self.memory:
                            self.memory.add_message(self.session_id, prompt, response)
                        return response
            except Exception as e:
                logger.error(f"Model generation failed: {e}, falling back to retrieval")

        logger.info("Using retrieval-based response")
        response = self._format_professional_response(context_result, inferred_task, prompt)
        logger.info(f"Generated response length: {len(response)}")
        
        if Memory and self.memory:
            self.memory.add_message(self.session_id, prompt, response)
        return response
    
    def get_relevant_context(self, query, max_context_length=1000):
        try:
            if not self.doc_texts:
                logger.warning("No documents available for context retrieval")
                return ""

            context_result = self._search_documents(query)
            if not context_result:
                return ""

            content = context_result.get("content", "")
            sentences = context_result.get("sentences", [])
            context = " ".join(sentences[:3]) if sentences else content[:max_context_length]
            return context[:max_context_length] + ("..." if len(context) > max_context_length else "")
        except Exception as e:
            logger.error(f"Error getting relevant context: {e}")
            return ""

    def _format_professional_response(self, context_result, task_type, prompt):
        content = context_result.get("content", "No relevant information found in the provided documents.")
        sources = context_result.get("sources", ["unknown"])
        confidence = context_result.get("confidence", 0.0)
        sentences = context_result.get("sentences", [])
        
        if task_type == "numerical":
            return self._format_numerical_response(content, sources, sentences, prompt)
        elif task_type == "creative":
            return self._format_creative_response(content, sources, sentences, prompt)
        elif task_type == "translate":
            return self._format_translate_response(content, sources, sentences, prompt)
        elif task_type in ["definition", "law"]:
            response = self._format_definition_response(content, sources, sentences, prompt)
        elif task_type == "explain":
            response = self._format_explanation_response(content, sources, sentences, prompt)
        elif task_type == "formula":
            response = self._format_formula_response(content, sources, sentences, prompt)
        elif task_type == "mcq":
            response = self._format_mcq_response(content, sources, sentences, prompt)
        elif task_type == "summary":
            response = self._format_summary_response(content, sources, sentences)
        elif task_type == "bullet_points":
            response = self._format_bullet_points_response(content, sources, sentences)
        else:
            response = self._format_general_response(content, sources, sentences, prompt)
        
        if content and content != "No relevant information found in the provided documents.":
            response += f"\n\n📚 *Source: {', '.join(sources)}*"
            response += f"\n*Confidence: {confidence:.2f}*"
        return response

    def _format_translate_response(self, content, sources, sentences, prompt):
        response = "🌐 **Translation**\n\n"
        response += f"Translation of document content is not supported, but here’s the relevant information:\n"
        response += f"**Content:** {sentences[0][:200] if sentences else content[:200]}...\n"
        response += f"\n📚 *Source: {', '.join(sources)}*"
        return response
    
    def _format_definition_response(self, content, sources, sentences, prompt):
        key_terms = re.findall(r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b', prompt) or [word for word in prompt.split() if word.lower() in prompt.lower()]
        main_term = key_terms[0] if key_terms else "concept"
        prompt_lower = prompt.lower()
        
        definition_patterns = [
            r'\b' + re.escape(main_term.lower()) + r'\s*(?:is defined as|states that|refers to|is given by)\s*([^.!?]+)',
            r'\b' + re.escape(main_term.lower()) + r'\s*[:=]\s*([^.!?]+)',
            r'\b' + re.escape(main_term.lower()) + r'\s*[A-Z]\s*=\s*[A-Z\d\s*/+-]+'
        ]
        definition_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for pattern in definition_patterns:
                match = re.search(pattern, sentence_lower, re.IGNORECASE)
                if match:
                    definition_sentences.append(sentence)
                    break
            else:
                if any(term in sentence_lower for term in prompt_lower.split()) or any(phrase in sentence_lower for phrase in ["law", "principle", "states", "defined"]):
                    definition_sentences.append(sentence)
        
        if not definition_sentences and sentences:
            definition_sentences = sentences[:2]
        
        response = f"📋 **{main_term}**\n\n"
        if definition_sentences:
            main_def = definition_sentences[0]
            response += f"**Definition:** {main_def.strip()}\n\n"
        else:
            response += f"**Definition:** {content[:200].strip()}...\n\n"
        
        if len(definition_sentences) > 1:
            response += f"**Key Points:**\n"
            for i, sentence in enumerate(definition_sentences[1:3], 1):
                response += f"{i}. {sentence.strip()}\n"
        
        math_patterns = re.findall(r'[A-Z]\s*=\s*[A-Z\d\s*/+-]+', content)
        if math_patterns:
            response += f"\n**Mathematical Form:** {math_patterns[0]}\n"
        
        example_sentences = [s for s in sentences if any(phrase in s.lower() for phrase in ["example", "application", "for instance"])]
        if example_sentences:
            response += f"\n**Example:** {example_sentences[0].strip()}\n"
        
        return response

    def _format_explanation_response(self, content, sources, sentences, prompt):
        response = f"💡 **Explanation**\n\n"
        
        if sentences:
            response += f"**Overview:** {sentences[0]}\n\n"
            response += f"**Details:**\n"
            for i, sentence in enumerate(sentences[1:4], 1):
                response += f"{i}. {sentence}\n"
        else:
            response += content[:500]
        
        return response

    def _format_formula_response(self, content, sources, sentences, prompt):
        response = f"🔢 **Formula/Equation**\n\n"
        
        math_patterns = re.findall(r'[A-Z]\s*=\s*[A-Z\d\s*/+-]+', content)
        response += f"**Mathematical Expression:** {math_patterns[0] if math_patterns else 'No formula found in the provided documents.'}\n\n"
        
        response += f"**Context:** {sentences[0] if sentences else 'No context available.'}\n\n"
        
        if len(sentences) > 1:
            response += f"**Application:** {sentences[1]}\n"
        
        return response

    def _format_mcq_response(self, content, sources, sentences, prompt):
        response = f"❓ **Multiple Choice Question**\n\n"
        
        if len(sentences) >= 4:
            response += f"**Question:** {sentences[0]}?\n\n"
            response += f"**Options:**\n"
            response += f"A) {sentences[1]}\n"
            response += f"B) {sentences[2]}\n"
            response += f"C) {sentences[3]}\n"
            response += f"D) None of the above\n\n"
        else:
            response += f"**Question Context:** {content[:300]}\n\n"
        
        return response

    def _format_summary_response(self, content, sources, sentences):
        response = f"📄 **Summary**\n\n"
        summary_text = " ".join(sentences[:3]) if sentences else content[:300]
        response += summary_text[:300] + ("..." if len(summary_text) > 300 else "")
        return response

    def _format_bullet_points_response(self, content, sources, sentences):
        response = f"📝 **Key Points**\n\n"
        for i, sentence in enumerate(sentences[:5], 1):
            response += f"• {sentence}\n"
        return response

    def _format_general_response(self, content, sources, sentences, prompt):
        response = f"💬 **Answer**\n\n"
        response += " ".join(sentences[:3]) if sentences else content[:400]
        response += ("..." if len(content) > 400 else "")
        return response

    def _post_process_response(self, text):
        text = text.replace('<UNK>', '').replace('<PAD>', '').replace('<SOS>', '').replace('<EOS>', '')
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'(.)\1{3,}', r'\1\1', text)
        words = text.split()
        filtered_words = [w for w in words if len(w) > 1 or w in ['a', 'I']]
        text = ' '.join(filtered_words)
        if len(filtered_words) < 3:
            return ""
        if text and not text.endswith(('.', '?', '!')):
            text += '.'
        return text

    def get_status(self):
        return {
            "trained": self.is_trained,
            "device": str(self.device),
            "documents": len(self.doc_texts),
            "tokenizer_vocab": len(self.tokenizer.token_to_id) if self.tokenizer.trained else 0,
            "embeddings": len(self.doc_embeddings)
        }

    def save_model(self, save_path=None, domain_name=None):
        try:
            if domain_name is None:
                if self.doc_texts:
                    first_filename = next(iter(self.doc_texts))
                    base_name = os.path.splitext(first_filename)[0]
                else:
                    base_name = "default"
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
                domain_name = f"model_{base_name}_{timestamp}"

            save_dir = Path("saved_models") / domain_name
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{domain_name}_checkpoint.pt"
            final_path = save_dir / filename
            tokenizer_path = save_dir / "uml_tokenizer.pkl"

            torch.save({
                'model_state_dict': self.model.state_dict(),
                'vocab_size': self.vocab_size,
                'doc_texts': self.doc_texts,
                'is_trained': self.is_trained
            }, final_path)

            self.tokenizer.save_pickle(tokenizer_path)

            models_trained = {}
            if os.path.exists(self.models_trained_path):
                try:
                    with open(self.models_trained_path, 'r', encoding='utf-8') as f:
                        models_trained = json.load(f)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load models_trained.json: {e}")

            model_metadata = {
                'domain': domain_name,
                'filename': filename,
                'path': str(final_path),
                'tokenizer_path': str(tokenizer_path),
                'timestamp': datetime.now().isoformat(),
                'vocab_size': self.vocab_size,
                'num_documents': len(self.doc_texts)
            }
            models_trained[filename] = model_metadata
            with open(self.models_trained_path, 'w', encoding='utf-8') as f:
                json.dump(models_trained, f, indent=2)

            logger.info(f"✅ Model and tokenizer saved successfully to {final_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")
            return False

    def load_model(self, model_path=None, tokenizer_path=None, domain_name=None):
        try:
            if model_path is None or tokenizer_path is None:
                if domain_name is None:
                    logger.warning("⚠️ No domain name or paths provided. Using default model.")
                    model_path = Path("saved_models/default/checkpoint.pt")
                    tokenizer_path = Path("saved_models/default/uml_tokenizer.pkl")
                else:
                    model_path = Path(f"saved_models/{domain_name}/{domain_name}_checkpoint.pt")
                    tokenizer_path = Path(f"saved_models/{domain_name}/uml_tokenizer.pkl")
    
            if not model_path.exists():
                logger.warning(f"❌ Model file {model_path} does not exist. Initializing new model.")
                self.model = RAGModel(vocab_size=self.vocab_size).to(self.device)
                self.model.tokenizer = self.tokenizer
                return False
    
            try:
                state = torch.load(model_path, map_location=self.device)
                state_dict = state.get("model_state_dict", state)
                self.model.load_state_dict(state_dict, strict=False)
                self.doc_texts = state.get("doc_texts", {})
                self.is_trained = state.get("is_trained", False)
                if "vocab_size" in state:
                    self.vocab_size = state["vocab_size"]
                logger.info(f"✅ Successfully loaded model from {model_path}")
                logger.info(f"📄 Loaded {len(self.doc_texts)} documents from checkpoint")
            except Exception as e:
                logger.error(f"❌ Failed to load model from {model_path}: {e}")
                self.model = RAGModel(vocab_size=self.vocab_size).to(self.device)
                self.model.tokenizer = self.tokenizer
                return False
    
            if not tokenizer_path.exists():
                logger.warning(f"❌ Tokenizer file {tokenizer_path} not found. Using default tokenizer.")
                return False
    
            try:
                self.tokenizer.load_pickle(str(tokenizer_path))
                self.model.tokenizer = self.tokenizer
                logger.info(f"✅ Tokenizer loaded from {tokenizer_path}")
            except Exception as e:
                logger.error(f"❌ Failed to load tokenizer: {e}")
                return False
    
            logger.info(f"✅ Model and tokenizer loaded successfully for domain: {domain_name or 'default'}")
            return True
        except Exception as e:
            logger.error(f"❌ Error during model loading: {e}")
            return False
    
    def clear_documents(self):
        self.doc_texts = {}
        self.doc_embeddings = {}
        logger.info("✅ Documents cleared")

    def get_document_summary(self, filename: str) -> str:
        if filename not in self.doc_texts:
            return f"❌ Document {filename} not found"
        doc = self.doc_texts[filename]
        sentences = doc.get("sentences", [])
        if not sentences:
            return "❌ No content available for summary"
        summary = " ".join(sentences[:3])
        return self._post_process_response(summary[:500] + "..." if len(summary) > 500 else summary)