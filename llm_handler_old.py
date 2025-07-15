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
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import hashlib
import pickle
import tempfile
from tokenizer.uml_tokenizer import UnigramTokenizer
from llm_components.RAGModel import RAGModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Helper functions (unchanged from your original code)
def format_memory_stats():
    try:
        import psutil
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
        "mcq": ["mcq", "multiple choice", "quiz", "questionnaire", "10 questions", "generate questions", "test questions"],
        "ppt": ["ppt", "slides", "presentation"],
        "summary": ["summarize", "overview", "in brief"],
        "bullet_points": ["bullet", "points", "outline", "key points"],
        "keywords": ["keywords", "terms", "key words"],
        "short_note": ["short note", "note on", "explain shortly"],
        "definition": ["define", "definition", "what is", "state"],
        "explain": ["explain", "describe", "how does", "what happens"],
        "formula": ["formula", "equation", "expression"],
        "law": ["law", "principle", "rule", "theorem"]
    }
    for task, keywords in task_keywords.items():
        if any(kw in prompt_lower for kw in keywords):
            return task
    return "faq"

class LLMHandler:
    def __init__(self, model_path='checkpoint.pt', tokenizer_path='uml_tokenizer.pkl', model_version=None):
        logger.info("Initializing LLMHandler")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Device: {self.device}")
        
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.trained_models_path = "trained_models.json"
        self.model_version = model_version
    
        self.vocab_size = 10000
        self.seq_len = 512
        self.pad_token_id = 0
        self.sos_token_id = 2
        self.eos_token_id = 3
        self.top_k = 3
    
        # Initialize tokenizer and model
        self.tokenizer = UnigramTokenizer()
        self.model = RAGModel(vocab_size=self.vocab_size).to(self.device)
        self.model.tokenizer = self.tokenizer  # Set tokenizer in RAGModel
    
        self.doc_texts = {}
        self.doc_embeddings = {}
        self.is_trained = False
    
        self._log_model_params()
        self._load_model(model_version)
        self._load_tokenizer()
        
    def _log_model_params(self):
        total, trainable = count_parameters(self.model)
        logger.info(f"📊 Model Parameters: Total = {total:,}, Trainable = {trainable:,}")

    def _load_model(self, domain_name=None):
        if domain_name is None:
            logger.warning("No domain name provided. Using default model.")
            path = f"saved_models/{domain_name}/{domain_name}_checkpoint.pt"
        else:
            path = f"saved_models/{domain_name}/{domain_name}_checkpoint.pt"

        if not os.path.exists(path):
            logger.info(f"Model file {path} does not exist, will train new model")
            return False

        try:
            state = torch.load(path, map_location=self.device)
            state_dict = state.get("model_state_dict", state)
            self.model.load_state_dict(state_dict)

            self.doc_texts = state.get("doc_texts", {})            

            #if not self.doc_texts:
            #    logger.warning("⚠️ No document texts loaded from checkpoint")
            #else:
            #    logger.info(f"📄 Loaded {len(self.doc_texts)} documents from checkpoint")

            self.is_trained = state.get("is_trained", False)
            if "vocab_size" in state:
                self.vocab_size = state["vocab_size"]

            logger.info(f"✅ Successfully loaded model from {path}")
            logger.info(f"📄 Loaded {len(self.doc_texts)} documents from checkpoint")
            return True

        except Exception as e:
            logger.warning(f"❌ Failed to load model from {path}: {e}")
            return False

    def _load_tokenizer(self, tokenizer_version=None):
        root_tokenizer_path = Path("uml_tokenizer.pkl")  # always from root
        if root_tokenizer_path.exists():
            try:
                self.tokenizer.load_pickle(str(root_tokenizer_path))
                self.model.tokenizer = self.tokenizer
                logger.info(f"✅ Tokenizer loaded from {root_tokenizer_path} "
                            f"{'(version: ' + str(tokenizer_version) + ')' if tokenizer_version else ''}.")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to load tokenizer: {e}", exc_info=True)
                raise RuntimeError("Tokenizer loading failed. Ensure 'uml_tokenizer.pkl' is valid.")
        else:
            logger.error("❌ Tokenizer file 'uml_tokenizer.pkl' not found in root.")
            raise FileNotFoundError("Tokenizer file 'uml_tokenizer.pkl' is required but missing.")

    def _train_tokenizer(self):
        if not self.doc_texts:
            logger.error("No documents available to train tokenizer")
            return
        texts = []
        for filename, doc in self.doc_texts.items():
            content = doc.get("content", "")
            if not isinstance(content, str) or not content.strip():
                logger.error(f"Invalid or empty content for {filename}: {content}")
                continue
            if len(content) < 100:
                logger.warning(f"Content for {filename} is too short ({len(content)} characters), preview: {content[:200]}...")
            else:
                logger.info(f"Collected {filename} with {len(content)} characters for tokenizer training")
            texts.append(content)
        if not texts:
            logger.error("No valid texts for tokenizer training")
            return
        try:
            self.tokenizer.train(texts, vocab_size=max(self.vocab_size, 20000))
            self.tokenizer.save_pickle(self.tokenizer_path)
            self.model.tokenizer = self.tokenizer
            logger.info(f"Tokenizer trained with vocab size: {len(self.tokenizer.token_to_id)}")
            if len(self.tokenizer.token_to_id) <= 4:
                logger.error(f"Tokenizer vocabulary too small: {len(self.tokenizer.token_to_id)} tokens")
        except Exception as e:
            logger.error(f"Tokenizer training failed: {e}")

    def index_documents(self, documents, force_reindex=False):
        if isinstance(documents, tuple):
            documents = [documents]
        if not documents:
            logger.error("No documents provided for indexing")
            return False
        indexed = 0
        for filename, content in documents:
            if not isinstance(content, str) or not content.strip():
                logger.error(f"Invalid content for {filename}: type={type(content)}, content='{content}'")
                continue
            content_hash = _content_hash(content)
            
            # FIX 1: Allow re-indexing even if hash matches when force_reindex=True
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
    
    def train_on_documents(self, epochs=20, batch_size=8, save_path=None, patience=3, domain_name=None):
        if not self.doc_texts:
            logger.error("No documents available for training")
            return False

        # Auto-generate domain name if not provided
        if domain_name is None:
            first_filename = next(iter(self.doc_texts))
            base_name = os.path.splitext(first_filename)[0]
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
            domain_name = f"model_{base_name}_{timestamp}"

        # Create save directory
        save_dir = os.path.join("saved_models", domain_name)
        os.makedirs(save_dir, exist_ok=True)

        # Update paths
        self.model_path = os.path.join(save_dir, f"{domain_name}_checkpoint.pt")
        self.tokenizer_path = os.path.join(save_dir, "uml_tokenizer.pkl")
        if save_path is None:
            save_path = self.model_path

        if not self.tokenizer.trained:
            self._train_tokenizer()
            if len(self.tokenizer.token_to_id) <= 4:
                logger.error(f"Tokenizer training failed, vocabulary size: {len(self.tokenizer.token_to_id)}")
                return False

        # Prepare training data
        train_pairs = []
        for fname, doc in self.doc_texts.items():
            content = doc.get("content", "")
            if not content.strip():
                logger.warning(f"Skipping empty content in {fname}, preview: {content[:200]}...")
                continue
            try:
                token_ids = self.tokenizer.encode_ids(content)
                if not token_ids:
                    logger.warning(f"No tokens generated for {fname}")
                    continue
                logger.info(f"Token IDs sample for {fname}: {token_ids[:10]}")
                for i in range(0, len(token_ids) - self.seq_len * 2, self.seq_len):
                    input_chunk = token_ids[i: i + self.seq_len]
                    output_chunk = token_ids[i + self.seq_len: i + self.seq_len * 2]
                    input_chunk += [self.pad_token_id] * (self.seq_len - len(input_chunk))
                    output_chunk += [self.pad_token_id] * (self.seq_len - len(output_chunk))
                    train_pairs.append((input_chunk, output_chunk))
            except Exception as e:
                logger.error(f"Error tokenizing {fname}: {e}")
        if not train_pairs:
            logger.error("No valid training pairs found")
            return False

        logger.info(f"Prepared {len(train_pairs)} training chunks")
        train_data, val_data = train_test_split(train_pairs, test_size=0.1, random_state=42)

        def create_dataset(data):
            queries = torch.tensor([q for q, _ in data], dtype=torch.long)
            answers = torch.tensor([a for _, a in data], dtype=torch.long)
            return torch.utils.data.TensorDataset(queries, answers)

        train_loader = torch.utils.data.DataLoader(create_dataset(train_data), batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(create_dataset(val_data), batch_size=batch_size)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)

        best_val_loss = float('inf')
        no_improvement = 0
        self.model.train()
        
        for epoch in range(epochs):
            total_train_loss = 0.0
            for batch_idx, (queries, answers) in enumerate(train_loader):
                queries, answers = queries.to(self.device), answers.to(self.device)
                optimizer.zero_grad()
                logits = self.model(queries, answers)
                loss = criterion(logits.reshape(-1, logits.size(-1)), answers.reshape(-1))
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                if batch_idx % 10 == 0:
                    logger.info(f"[Epoch {epoch+1}] Batch {batch_idx} - Train Loss: {loss.item():.4f}")
            avg_train_loss = total_train_loss / len(train_loader)

            self.model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for queries, answers in val_loader:
                    queries, answers = queries.to(self.device), answers.to(self.device)
                    logits = self.model(queries, answers)
                    val_loss = criterion(logits.view(-1, logits.size(-1)), answers.view(-1))
                    total_val_loss += val_loss.item()
            avg_val_loss = total_val_loss / len(val_loader)

            logger.info(f"[Epoch {epoch+1}] ✅ Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improvement = 0

                # Save model with documents included
                try:
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'vocab_size': self.vocab_size,
                        'doc_texts': self.doc_texts  # FIX 2: Always save doc_texts
                    }, save_path)
                    logger.info(f"✅ New best model saved to {save_path}")
                except Exception as e:
                    logger.error(f"❌ Error saving checkpoint: {e}")
                    return False
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # FIX 3: Set is_trained to True after successful training
        self.is_trained = True
        logger.info("✅ Training complete")
        return True

    def generate_response(self, prompt, task="answer", max_length=10000, temperature=0.8, context=None):
        # Log initial state
        logger.info(f"Generating response for prompt: {prompt[:50]}...")
        logger.info(f"Documents available: {len(self.doc_texts)}")
        logger.info(f"Model trained: {self.is_trained}")
        logger.info(f"Tokenizer trained: {self.tokenizer.trained}")

        # Check if documents are available
        if not self.doc_texts:
            logger.warning("No documents available for response generation")
            return "Please upload documents first to enable response generation."

        # Infer task type based on prompt
        inferred_task = infer_task_type(prompt)
        logger.info(f"Inferred task type: {inferred_task}")

        # Get relevant context
        context_result = self._search_documents(prompt)

        if not context_result:
            return "I couldn't find relevant information in the documents."

        # Generate professional response
        response = self._format_professional_response(context_result, inferred_task, prompt)
    
        # Retrieval-based fallback if model isn't trained
        if not self.is_trained:
            logger.info("Model not trained, using retrieval-based response")
            context_result = self._faq_chunk_search(prompt)
            if context_result and context_result.get("confidence", 0) > 0.1:
                response = self._format_response(context_result, inferred_task, prompt)
                response += f"\n\nConfidence: {context_result['confidence']:.2f}"
                response += f"\nSource: {context_result['source']}"
                return response
            else:
                logger.warning("Low confidence in search results, attempting fallback")
                if context_result:
                    response = self._format_response(context_result, inferred_task, prompt)
                    response += f"\n\nConfidence: {context_result['confidence']:.2f}"
                    response += f"\nSource: {context_result['source']}"
                    return response
                # Last resort: return sample text from documents
                first_doc = next(iter(self.doc_texts.values()))
                sentences = first_doc.get("sentences", [])
                if sentences:
                    sample_text = " ".join(sentences[:3])
                    return self._format_response({"summary": sample_text[:300], "source": "unknown"}, inferred_task, prompt)
                return "I couldn't find relevant information in the documents to answer your question."

        # Check if tokenizer is trained
        if not self.tokenizer.trained:
            logger.warning("Tokenizer not trained, using retrieval fallback")
            context_result = self._faq_chunk_search(prompt)
            if context_result:
                return self._format_response(context_result, inferred_task, prompt)
            return "Please train the tokenizer first."

        # Encode prompt with error handling
        try:
            prompt_ids = self.tokenizer.encode_ids(prompt)
            if not isinstance(prompt_ids, list) or not all(isinstance(i, int) for i in prompt_ids):
                raise ValueError(f"Unexpected prompt_ids format: {type(prompt_ids)} -> {prompt_ids}")
            if not prompt_ids:
                logger.warning("Empty prompt encoding, using retrieval fallback")
                context_result = self._faq_chunk_search(prompt)
                if context_result:
                    return self._format_response(context_result, inferred_task, prompt)
                return "Could not encode the prompt. Please check the input."
        except Exception as e:
            logger.error(f"Error encoding prompt: {e}")
            context_result = self._faq_chunk_search(prompt)
            if context_result:
                return self._format_response(context_result, inferred_task, prompt)
            return "Error processing the prompt."

        # Prepare prompt tensor
        prompt_ids = prompt_ids[:self.seq_len]
        prompt_ids += [self.pad_token_id] * (self.seq_len - len(prompt_ids))
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        # Retrieve relevant context for augmentation
        context_result = self._faq_chunk_search(prompt)
        relevant_info = ""
        if context_result and context_result.get("confidence", 0) > 0.1:
            relevant_info = context_result["summary"]
            logger.info(f"Using context from {context_result['source']} with confidence {context_result['confidence']:.2f}")

        # Generate response using the model
        try:
            with torch.no_grad():
                # Call the model's generate method with standard signature
                generated_ids = self.model.generate(
                    input_ids=prompt_tensor,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=50
                )

                if generated_ids.numel() > 0:
                    generated_text = self.tokenizer.decode(generated_ids[0].tolist())
                    generated_text = self._post_process_response(generated_text)
                    response = self._format_response(
                        {"summary": generated_text, "source": context_result['source'] if context_result else "model"},
                        inferred_task,
                        prompt
                    )
                    if context_result and relevant_info:
                        response += f"\n\nContext Used: {relevant_info[:200]}..."
                        response += f"\nConfidence: {context_result['confidence']:.2f}"
                        response += f"\nSource: {context_result['source']}"
                    return response
                else:
                    # Fallback to retrieval if generation fails
                    if context_result:
                        return self._format_response(context_result, inferred_task, prompt)
                    return "Model failed to generate a response. Please try a different prompt."
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            # Fallback to retrieval
            if context_result:
                return self._format_response(context_result, inferred_task, prompt)
            return "Error generating response. Please try again."
    def _format_response(self, context_result, task_type, prompt):
        summary = context_result.get("summary", "")
        source = context_result.get("source", "unknown")
        response = ""

        if task_type == "mcq":
            sentences = summary.split(". ")
            if len(sentences) >= 4:
                response = "Multiple Choice Question:\n"
                response += f"Q: {sentences[0].strip()}?\n"
                response += f"A) {sentences[1].strip()}\n"
                response += f"B) {sentences[2].strip()}\n"
                response += f"C) {sentences[3].strip()}\n"
                response += "D) None of the above\n"
                response += f"(Source: {source})"
            else:
                response = f"Based on the documents: {summary}\n(Source: {source})"
        elif task_type == "ppt":
            sentences = summary.split(". ")
            response = "Presentation Slide Content:\n"
            response += "Title: Key Information\n"
            for i, sentence in enumerate(sentences[:3], 1):
                response += f"• Point {i}: {sentence.strip()}\n"
            response += f"(Source: {source})"
        elif task_type == "summary":
            response = f"Summary: {summary[:200]}{'...' if len(summary) > 200 else ''}\n(Source: {source})"
        elif task_type == "bullet_points":
            sentences = summary.split(". ")
            response = "Key Points:\n"
            for sentence in sentences[:3]:
                response += f"• {sentence.strip()}\n"
            response += f"(Source: {source})"
        elif task_type == "keywords":
            words = summary.lower().split()
            keywords = [w for w in set(words) if len(w) > 3][:5]
            response = "Keywords: " + ", ".join(keywords) + f"\n(Source: {source})"
        elif task_type == "short_note":
            response = f"Short Note: {summary[:150]}{'...' if len(summary) > 150 else ''}\n(Source: {source})"
        elif task_type == "definition":
            response = f"Definition: {summary[:100]}{'...' if len(summary) > 100 else ''}\n(Source: {source})"
        else:
            response = f"Answer: {summary}\n(Source: {source})"

        return response     
    def _post_process_response(self, text):
        text = text.replace('<UNK>', '').replace('<PAD>', '').replace('< SOS >', '').replace('<EOS>', '')
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'(.)\1{3,}', r'\1\1', text)
        words = text.split()
        filtered_words = [w for w in words if len(w) > 1 or w in ['a', 'I']]
        text = ' '.join(filtered_words)
        if len(filtered_words) < 3 or sum(c.isalpha() for c in text) / max(len(text), 1) < 0.7:
            return "I'm sorry, I couldn't understand that. Could you please rephrase?"
        if text:
            text = text[0].upper() + text[1:]
        if text and not text.endswith(('.', '?', '!')):
            text += '.'
        return text

    def _faq_chunk_search(self, prompt: str, top_k: int = None):
        top_k = top_k or self.top_k
        if not self.doc_texts:
            return None
        all_chunks = []
        chunk_sources = []
        for fname, doc in self.doc_texts.items():
            sentences = doc.get("sentences", [])
            for sentence in sentences:
                if len(sentence.strip()) > 40:
                    cleaned_sentence = re.sub(r'\(cid\d+\)', '', sentence)
                    cleaned_sentence = re.sub(r'[^\w\s,.!?]', '', cleaned_sentence)
                    cleaned_sentence = re.sub(r'\s+', ' ', cleaned_sentence).strip()
                    if cleaned_sentence and len(cleaned_sentence) > 20:
                        all_chunks.append(cleaned_sentence)
                        chunk_sources.append(fname)
        if not all_chunks:
            return None
        prompt_embedding = get_embedding(prompt)
        chunk_embeddings = [get_embedding(chunk) for chunk in all_chunks]
        similarities = []
        for chunk_emb in chunk_embeddings:
            try:
                if len(prompt_embedding) != len(chunk_emb):
                    max_len = max(len(prompt_embedding), len(chunk_emb))
                    prompt_embedding = np.pad(prompt_embedding, (0, max_len - len(prompt_embedding)))
                    chunk_emb = np.pad(chunk_emb, (0, max_len - len(chunk_emb)))
                sim = cosine_similarity([prompt_embedding], [chunk_emb])[0][0]
                similarities.append(sim)
            except:
                similarities.append(0.0)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_chunks = [all_chunks[i] for i in top_indices if similarities[i] > 0.1]
        if not top_chunks:
            return None
        combined_text = " ".join(top_chunks)
        avg_confidence = np.mean([similarities[i] for i in top_indices if similarities[i] > 0.1])
        sources = list(set(chunk_sources[i] for i in top_indices if similarities[i] > 0.1))
        return {
            "summary": combined_text[:500] + "..." if len(combined_text) > 500 else combined_text,
            "confidence": avg_confidence,
            "source": ", ".join(sources)
        }
    
    def is_retriever_ready(self) -> bool:
        return bool(self.doc_texts)
    
    def get_relevant_context(self, query, max_context_length=1000):
        """Get relevant context from indexed documents"""
        try:
            if not self.doc_texts:
                return ""

            # Simple keyword matching for context retrieval
            query_words = set(query.lower().split())
            relevant_chunks = []

            for doc_name, doc_info in self.doc_texts.items():
                content = doc_info.get("content", "")

                # Split content into chunks
                chunks = [content[i:i+200] for i in range(0, len(content), 200)]

                for chunk in chunks:
                    chunk_words = set(chunk.lower().split())
                    # Calculate overlap
                    overlap = len(query_words.intersection(chunk_words))
                    if overlap > 0:
                        relevant_chunks.append((chunk, overlap))

            # Sort by relevance and take top chunks
            relevant_chunks.sort(key=lambda x: x[1], reverse=True)

            context = ""
            for chunk, score in relevant_chunks[:3]:  # Top 3 chunks
                context += chunk + "\n\n"
                if len(context) > max_context_length:
                    break
                
            return context[:max_context_length]

        except Exception as e:
            logger.error(f"Error getting relevant context: {e}")
            return ""
    def search_documents(self, query):
        """Search through documents for relevant information"""
        try:
            if not self.doc_texts:
                return "No documents available to search."

            query_words = set(query.lower().split())
            best_matches = []

            for doc_name, doc_info in self.doc_texts.items():
                content = doc_info.get("content", "")

                # Split into sentences
                sentences = content.split('.')

                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) < 20:  # Skip very short sentences
                        continue

                    sentence_words = set(sentence.lower().split())
                    overlap = len(query_words.intersection(sentence_words))

                    if overlap > 0:
                        best_matches.append((sentence, overlap, doc_name))

            # Sort by relevance
            best_matches.sort(key=lambda x: x[1], reverse=True)

            if not best_matches:
                return "I couldn't find relevant information in the documents for your query."

            # Create response from top matches
            response = f"Based on the documents, here's what I found:\n\n"

            for i, (sentence, score, doc_name) in enumerate(best_matches[:3]):
                response += f"{i+1}. {sentence.strip()}\n"
                if i < 2:  # Add source for first few results
                    response += f"   (Source: {doc_name})\n\n"

            return response

        except Exception as e:
            logger.error(f"Error in search_documents: {e}")
            return "I encountered an error while searching the documents."
    def _search_documents(self, query):
        """Enhanced document search with better relevance scoring"""
        if not self.doc_texts:
            return None
            
        all_chunks = []
        chunk_sources = []
        
        for fname, doc in self.doc_texts.items():
            sentences = doc.get("sentences", [])
            for sentence in sentences:
                if len(sentence.strip()) > 40:
                    # Clean and normalize sentence
                    cleaned = re.sub(r'\(cid\d+\)', '', sentence)
                    cleaned = re.sub(r'[^\w\s,.!?()-]', '', cleaned)
                    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                    
                    if cleaned and len(cleaned) > 20:
                        all_chunks.append(cleaned)
                        chunk_sources.append(fname)
        
        if not all_chunks:
            return None
        
        # Calculate similarities
        query_embedding = get_embedding(query)
        similarities = []
        
        for chunk in all_chunks:
            chunk_embedding = get_embedding(chunk)
            try:
                if len(query_embedding) != len(chunk_embedding):
                    max_len = max(len(query_embedding), len(chunk_embedding))
                    query_embedding = np.pad(query_embedding, (0, max_len - len(query_embedding)))
                    chunk_embedding = np.pad(chunk_embedding, (0, max_len - len(chunk_embedding)))
                
                sim = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
                similarities.append(sim)
            except:
                similarities.append(0.0)
        
        # Get top results
        top_indices = np.argsort(similarities)[-self.top_k:][::-1]
        top_chunks = [all_chunks[i] for i in top_indices if similarities[i] > 0.05]
        
        if not top_chunks:
            return None
        
        combined_text = " ".join(top_chunks)
        avg_confidence = np.mean([similarities[i] for i in top_indices if similarities[i] > 0.05])
        sources = list(set(chunk_sources[i] for i in top_indices if similarities[i] > 0.05))
        
        return {
            "content": combined_text,
            "confidence": avg_confidence,
            "sources": sources,
            "chunks": top_chunks
        }

    def _format_professional_response(self, context_result, task_type, prompt):
        """Format response professionally based on task type"""
        content = context_result.get("content", "")
        sources = context_result.get("sources", [])
        confidence = context_result.get("confidence", 0)
        
        if task_type == "definition" or task_type == "law":
            return self._format_definition_response(content, sources, prompt)
        elif task_type == "explain":
            return self._format_explanation_response(content, sources, prompt)
        elif task_type == "formula":
            return self._format_formula_response(content, sources, prompt)
        elif task_type == "mcq":
            return self._format_mcq_response(content, sources, prompt)
        elif task_type == "summary":
            return self._format_summary_response(content, sources)
        elif task_type == "bullet_points":
            return self._format_bullet_points_response(content, sources)
        else:
            return self._format_general_response(content, sources, prompt)

    def _format_definition_response(self, content, sources, prompt):
        """Format definition/law responses professionally"""
        # Extract key terms from prompt
        key_terms = re.findall(r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b', prompt)
        main_term = key_terms[0] if key_terms else "concept"
        
        # Find definition-like sentences
        sentences = content.split('.')
        definition_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(word in sentence.lower() for word in ['states', 'law', 'principle', 'defined', 'equation']):
                definition_sentences.append(sentence)
        
        if not definition_sentences:
            definition_sentences = sentences[:2]
        
        response = f"📋 **{main_term}**\n\n"
        
        # Main definition
        main_def = definition_sentences[0] if definition_sentences else content[:200]
        response += f"**Definition:** {main_def.strip()}\n\n"
        
        # Additional context if available
        if len(definition_sentences) > 1:
            response += f"**Key Points:**\n"
            for i, sentence in enumerate(definition_sentences[1:3], 1):
                response += f"{i}. {sentence.strip()}\n"
        
        # Mathematical representation if found
        math_patterns = re.findall(r'[A-Z]\s*=\s*[A-Z\d\s*/+-]+', content)
        if math_patterns:
            response += f"\n**Mathematical Form:** {math_patterns[0]}\n"
        
        response += f"\n📚 *Source: {', '.join(sources)}*"
        return response

    def _format_explanation_response(self, content, sources, prompt):
        """Format explanatory responses"""
        response = f"💡 **Explanation**\n\n"
        
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        
        if len(sentences) >= 3:
            response += f"**Overview:** {sentences[0]}\n\n"
            response += f"**Details:**\n"
            for i, sentence in enumerate(sentences[1:4], 1):
                response += f"{i}. {sentence}\n"
        else:
            response += content[:500]
        
        response += f"\n📚 *Source: {', '.join(sources)}*"
        return response

    def _format_formula_response(self, content, sources, prompt):
        """Format formula/equation responses"""
        response = f"🔢 **Formula/Equation**\n\n"
        
        # Extract mathematical expressions
        math_patterns = re.findall(r'[A-Z]\s*=\s*[A-Z\d\s*/+-]+', content)
        
        if math_patterns:
            response += f"**Mathematical Expression:** {math_patterns[0]}\n\n"
        
        # Add context
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        response += f"**Context:** {sentences[0] if sentences else 'Formula derivation'}\n\n"
        
        if len(sentences) > 1:
            response += f"**Application:** {sentences[1]}\n"
        
        response += f"\n📚 *Source: {', '.join(sources)}*"
        return response

    def _format_mcq_response(self, content, sources, prompt):
        """Format MCQ responses"""
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        
        if len(sentences) >= 4:
            response = f"❓ **Multiple Choice Question**\n\n"
            response += f"**Question:** {sentences[0]}?\n\n"
            response += f"**Options:**\n"
            response += f"A) {sentences[1]}\n"
            response += f"B) {sentences[2]}\n"
            response += f"C) {sentences[3]}\n"
            response += f"D) None of the above\n\n"
            response += f"📚 *Source: {', '.join(sources)}*"
        else:
            response = f"❓ **Question Context**\n\n{content}\n\n📚 *Source: {', '.join(sources)}*"
        
        return response

    def _format_summary_response(self, content, sources):
        """Format summary responses"""
        response = f"📄 **Summary**\n\n"
        
        # Take first 300 characters for summary
        summary_text = content[:300]
        if len(content) > 300:
            summary_text += "..."
        
        response += summary_text
        response += f"\n\n📚 *Source: {', '.join(sources)}*"
        return response

    def _format_bullet_points_response(self, content, sources):
        """Format bullet points responses"""
        response = f"📝 **Key Points**\n\n"
        
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        
        for i, sentence in enumerate(sentences[:5], 1):
            response += f"• {sentence}\n"
        
        response += f"\n📚 *Source: {', '.join(sources)}*"
        return response

    def _format_general_response(self, content, sources, prompt):
        """Format general responses"""
        response = f"💬 **Answer**\n\n"
        response += content[:400]
        if len(content) > 400:
            response += "..."
        response += f"\n\n📚 *Source: {', '.join(sources)}*"
        return response

    def _enhance_with_generation(self, prompt, base_response, max_length, temperature):
        """Enhance response with model generation"""
        try:
            prompt_ids = self.tokenizer.encode_ids(prompt)
            if not prompt_ids:
                return base_response
            
            prompt_ids = prompt_ids[:self.seq_len]
            prompt_ids += [self.pad_token_id] * (self.seq_len - len(prompt_ids))
            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=prompt_tensor,
                    max_length=min(max_length, 200),
                    temperature=temperature,
                    top_k=50
                )
                
                if generated_ids.numel() > 0:
                    generated_text = self.tokenizer.decode(generated_ids[0].tolist())
                    generated_text = self._post_process_response(generated_text)
                    
                    if len(generated_text.strip()) > 50:
                        return f"{base_response}\n\n**Enhanced Details:**\n{generated_text}"
                        
        except Exception as e:
            logger.error(f"Generation enhancement failed: {e}")
        
        return base_response

    def _post_process_response(self, text):
        """Clean and post-process generated text"""
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

    def save_model(self, save_path=None, domain_name=None):
        """Save model with metadata"""
        try:
            if domain_name is None:
                if self.doc_texts:
                    first_filename = next(iter(self.doc_texts))
                    base_name = os.path.splitext(first_filename)[0]
                else:
                    base_name = "default"
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
                domain_name = f"model_{base_name}_{timestamp}"

            save_dir = os.path.join("saved_models", domain_name)
            os.makedirs(save_dir, exist_ok=True)
            final_path = os.path.join(save_dir, f"{domain_name}_checkpoint.pt")

            torch.save({
                'model_state_dict': self.model.state_dict(),
                'vocab_size': self.vocab_size,
                'doc_texts': self.doc_texts,
                'is_trained': self.is_trained
            }, final_path)
            
            self.tokenizer.save_pickle(os.path.join(save_dir, "uml_tokenizer.pkl"))
            logger.info(f"✅ Model saved to {final_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")
            return False
            
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

            save_dir = os.path.join("saved_models", domain_name)
            os.makedirs(save_dir, exist_ok=True)
            filename = f"{domain_name}_checkpoint.pt"
            final_path = os.path.join(save_dir, filename)

            torch.save({
                'model_state_dict': self.model.state_dict(),
                'vocab_size': self.vocab_size,
                'doc_texts': self.doc_texts,  # Always save documents
                'is_trained': self.is_trained
            }, final_path)
            
            self.tokenizer.save_pickle(os.path.join(save_dir, "uml_tokenizer.pkl"))

            trained_models_path = "trained_models.json"
            trained_models = {}
            if os.path.exists(trained_models_path):
                try:
                    with open(trained_models_path, 'r', encoding='utf-8') as f:
                        trained_models = json.load(f)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load trained_models.json: {e}")

            model_metadata = {
                'domain': domain_name,
                'filename': filename,
                'path': final_path,
                'timestamp': datetime.now().isoformat(),
                'vocab_size': self.vocab_size,
                'num_documents': len(self.doc_texts)
            }
            trained_models[filename] = model_metadata
            with open(trained_models_path, 'w', encoding='utf-8') as f:
                json.dump(trained_models, f, indent=2)

            logger.info(f"✅ Model and tokenizer saved successfully to {final_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")
            return False

    def load_model(self, model_path=None, tokenizer_path=None):
        try:
            if model_path is None or tokenizer_path is None:
                logger.warning("⚠️ Model path or tokenizer path not provided. Loading default model.")
                model_path = "saved_models/default/default_checkpoint.pt"
                tokenizer_path = "saved_models/default/uml_tokenizer.pkl"

            self.model_path = model_path
            self.tokenizer_path = tokenizer_path

            domain_name = Path(model_path).parent.name
            model_loaded = self._load_model(domain_name=domain_name)
            tokenizer_loaded = self._load_tokenizer()
            if model_loaded or tokenizer_loaded:
                logger.info("✅ Model and tokenizer loaded successfully")
            else:
                logger.warning("⚠️ Model and/or tokenizer could not be loaded properly")
        except Exception as e:
            logger.error(f"❌ Error during loading: {e}")

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
        summary_length = min(3, len(sentences))
        summary = " ".join(sentences[:summary_length])
        return self._post_process_response(summary[:500] + "..." if len(summary) > 500 else summary)