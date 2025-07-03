import torch
import torch.nn as nn
from datetime import datetime
import os
import re
import logging
from pythonjsonlogger import jsonlogger
import streamlit as st
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import hashlib
import time
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Helper functions
def format_memory_stats():
    """Format memory statistics for logging"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return f"Memory: {memory.percent}% used ({memory.used/1024/1024/1024:.1f}GB/{memory.total/1024/1024/1024:.1f}GB)"
    except:
        return "Memory stats unavailable"

def log_gpu_stats(logger):
    """Log GPU statistics"""
    if torch.cuda.is_available():
        logger.info(f"GPU Memory: {torch.cuda.memory_allocated()/1024/1024/1024:.1f}GB allocated")
    else:
        logger.info("No GPU available")

def get_embedding(text: str):
    """Simple embedding function using TF-IDF"""
    try:
        # Simple TF-IDF embedding for fallback
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        # Need at least 2 documents for TF-IDF, so we add a dummy document
        corpus = [text, "dummy document"]
        embeddings = vectorizer.fit_transform(corpus)
        return embeddings[0].toarray().flatten()
    except:
        # Fallback to simple word count vector
        words = text.lower().split()
        return np.array([len(words), len(set(words)), len(text)])

# -----------------------------
# ✅ Improved BPE Tokenizer (Simple Implementation)
# -----------------------------
class ImprovedBPETokenizer:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inv_vocab = {}
        self.word_freq = {}
        self.trained = False
        
    def train(self, texts):
        """Train the tokenizer on a list of texts"""
        if not texts:
            return
            
        # Simple word-based tokenization for now
        all_words = []
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)
        
        # Count word frequencies
        from collections import Counter
        word_counts = Counter(all_words)
        
        # Build vocabulary with most frequent words
        special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        self.vocab = {token: i for i, token in enumerate(special_tokens)}
        
        # Add most frequent words
        for word, count in word_counts.most_common(self.vocab_size - len(special_tokens)):
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        
        # Create inverse vocabulary
        self.inv_vocab = {i: word for word, i in self.vocab.items()}
        self.trained = True
    
    def encode(self, text):
        """Encode text to token IDs"""
        if not self.trained:
            return [1]  # Return UNK token
        
        words = re.findall(r'\b\w+\b', text.lower())
        return [self.vocab.get(word, 1) for word in words]  # 1 is UNK
    
    def decode(self, token_ids):
        """Decode token IDs to text"""
        if not self.trained:
            return "untrained tokenizer"
        
        words = [self.inv_vocab.get(id, '<UNK>') for id in token_ids]
        return ' '.join(words)
    
    def save(self, path):
        """Save tokenizer to file"""
        data = {
            'vocab': self.vocab,
            'vocab_size': self.vocab_size,
            'trained': self.trained
        }
        with open(path, 'wb') as f:
            import pickle
            pickle.dump(data, f)
    
    def load(self, path):
        """Load tokenizer from file"""
        with open(path, 'rb') as f:
            import pickle
            data = pickle.load(f)
        
        self.vocab = data['vocab']
        self.vocab_size = data['vocab_size']
        self.trained = data['trained']
        self.inv_vocab = {i: word for word, i in self.vocab.items()}

# -----------------------------
# ✅ Enhanced RAG Model with Better Generation
# -----------------------------
class RAGModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(embed_dim, hidden_dim * 2, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(0.1)
        
        # Attention mechanism for better generation
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.context_projection = nn.Linear(hidden_dim * 2, embed_dim)

    def forward(self, input_ids, target_ids=None, context_embeddings=None):
        embedded_input = self.embedding(input_ids)
        encoder_output, (hidden, cell) = self.encoder(embedded_input)
        
        # Merge bidirectional states
        hidden = self._merge_bidir(hidden)
        cell = self._merge_bidir(cell)

        if target_ids is not None:
            # Training mode
            embedded_target = self.embedding(target_ids)
            
            # Apply attention if context is provided
            if context_embeddings is not None:
                context_proj = self.context_projection(context_embeddings)
                embedded_target, _ = self.attention(embedded_target, context_proj, context_proj)
            
            decoder_output, _ = self.decoder(embedded_target, (hidden, cell))
            decoder_output = self.dropout(decoder_output)
            logits = self.output_layer(decoder_output)
            return logits
        else:
            # Inference mode - return encoder output for retrieval
            pooled = torch.mean(encoder_output, dim=1)
            return pooled

    def encode_document(self, input_ids):
        """Encode document for retrieval"""
        embedded = self.embedding(input_ids)
        output, _ = self.encoder(embedded)
        pooled = torch.mean(output, dim=1)
        return pooled

    def generate_with_context(self, input_ids, context_embeddings=None, max_length=100, temperature=0.8, top_k=50):
        """Enhanced generation with context and sampling"""
        self.eval()
        with torch.no_grad():
            # Get encoder states
            embedded_input = self.embedding(input_ids)
            encoder_output, (hidden, cell) = self.encoder(embedded_input)
            hidden = self._merge_bidir(hidden)
            cell = self._merge_bidir(cell)
            
            # Start with SOS token
            current_input = torch.tensor([[2]], dtype=torch.long, device=input_ids.device)  # SOS token
            generated = []
            
            for _ in range(max_length):
                embedded = self.embedding(current_input)
                
                # Apply attention if context is provided
                if context_embeddings is not None:
                    context_proj = self.context_projection(context_embeddings)
                    embedded, _ = self.attention(embedded, context_proj, context_proj)
                
                output, (hidden, cell) = self.decoder(embedded, (hidden, cell))
                logits = self.output_layer(output)
                
                # Apply temperature and top-k sampling
                logits = logits / temperature
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                    probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
                    next_token_idx = torch.multinomial(probs.squeeze(), 1)
                    next_token = top_k_indices.squeeze()[next_token_idx]
                else:
                    next_token = torch.multinomial(torch.nn.functional.softmax(logits.squeeze(), dim=-1), 1)
                
                token_id = next_token.item()
                if token_id == 3:  # EOS token
                    break
                    
                generated.append(token_id)
                current_input = next_token.unsqueeze(0)
            
            return torch.tensor([generated])

    def _merge_bidir(self, h):
        """Merge bidirectional LSTM states"""
        h = h.view(self.num_layers, 2, h.size(1), h.size(2))
        return torch.cat([h[:, 0], h[:, 1]], dim=-1)

# -----------------------------
# ✅ Enhanced LLM Handler with Model-Based Generation
# -----------------------------
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
        "definition": ["define", "definition", "what is"],
    }
    for task, keywords in task_keywords.items():
        if any(kw in prompt_lower for kw in keywords):
            return task
    return "faq"

class LLMHandler:
    def __init__(self, model_path='checkpoint.pt', tokenizer_path='bpe_tokenizer.pkl', model_version=None):
        """Initialize LLMHandler with optional model version"""
        logger.info("Initializing LLMHandler")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Device: {self.device}")
        
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.trained_models_path = "trained_models.json"
        self.model_version = model_version
    
        self.vocab_size = 5000
        self.seq_len = 512
        self.pad_token_id = 0
        self.top_k = 3
    
        # Initialize tokenizer and model
        self.tokenizer = ImprovedBPETokenizer(vocab_size=self.vocab_size)
        self.model = RAGModel(vocab_size=self.vocab_size).to(self.device)
    
        self.doc_texts = {}
        self.doc_embeddings = {}
        self.is_trained = False
    
        self._log_model_params()
        self._load_tokenizer(model_version)
        self._load_model(model_version)

    def _log_model_params(self):
        total, trainable = count_parameters(self.model)
        logger.info(f"📊 Model Parameters: Total = {total:,}, Trainable = {trainable:,}")

    def _load_model(self, model_version=None):
        if Path(self.model_path).exists():
            try:
                state = torch.load(self.model_path, map_location=self.device)
                if 'model_state_dict' in state:
                    self.model.load_state_dict(state['model_state_dict'])
                else:
                    self.model.load_state_dict(state)
                self.is_trained = True
                logger.info(f"✅ Model loaded successfully{' (version: ' + str(model_version) + ')' if model_version else ''}.")
            except Exception as e:
                logger.warning(f"⚠️ Model corrupted: {e}. Will retrain when needed.")
                self.is_trained = False
        else:
            logger.info("No saved model found. Will train when documents are available.")
            self.is_trained = False

    def _load_tokenizer(self, tokenizer_version=None):
        """Load tokenizer with optional version parameter"""
        if Path(self.tokenizer_path).exists():
            try:
                self.tokenizer.load(self.tokenizer_path)
                logger.info(f"✅ Tokenizer loaded successfully{' (version: ' + str(tokenizer_version) + ')' if tokenizer_version else ''}.")
            except Exception as e:
                logger.warning(f"⚠️ Tokenizer corrupted: {e}. Will retrain when needed.")
                self.tokenizer = ImprovedBPETokenizer(vocab_size=self.vocab_size)
        else:
            logger.info("No saved tokenizer found. Will train when documents are available.")

    def _train_tokenizer(self):
        if not self.doc_texts:
            logger.warning("No documents available to train tokenizer.")
            return
        
        texts = []
        for doc in self.doc_texts.values():
            if isinstance(doc, dict):
                content = doc.get("content", "")
            else:
                content = str(doc)
            if content:
                texts.append(content)
        
        if texts:
            self.tokenizer.train(texts)
            self.tokenizer.save(self.tokenizer_path)
            logger.info("✅ Tokenizer trained and saved.")

    def index_documents(self, documents):
        """Index documents for retrieval"""
        if isinstance(documents, tuple):
            documents = [documents]
        
        if not documents:
            logger.warning("No documents provided for indexing")
            return False

        indexed = 0
        for filename, content in documents:
            if not isinstance(content, str) or not content.strip():
                logger.warning(f"Skipping invalid content in {filename}")
                continue
                
            content_hash = _content_hash(content)
            if filename in self.doc_texts:
                existing_hash = self.doc_texts[filename].get("hash", "")
                if existing_hash == content_hash:
                    logger.info(f"Skipping already indexed document: {filename}")
                    continue
            
            # Process content into sentences for better retrieval
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            
            self.doc_texts[filename] = {
                "content": content,
                "sentences": sentences,
                "hash": content_hash
            }
            logger.info(f"Indexed document: {filename}")
            indexed += 1

        # Train tokenizer if we have new documents
        if indexed > 0 and not self.tokenizer.trained:
            self._train_tokenizer()

        logger.info(f"Indexing complete: {indexed} documents indexed")
        return indexed > 0

    def train_on_documents(self, epochs=5, batch_size=8, save_path="checkpoint.pt"):
        """Train the model on indexed documents"""
        if not self.doc_texts:
            logger.warning("No documents available for training.")
            return

        logger.info(f"🚀 Starting training on {len(self.doc_texts)} documents")
        
        # Ensure tokenizer is trained
        if not self.tokenizer.trained:
            self._train_tokenizer()

        # Create training pairs from sentences
        train_pairs = []
        for doc in self.doc_texts.values():
            sentences = doc.get("sentences", [])
            for i in range(len(sentences) - 1):
                if len(sentences[i]) > 30 and len(sentences[i + 1]) > 30:
                    train_pairs.append((sentences[i], sentences[i + 1]))

        if not train_pairs:
            logger.warning("No training pairs found.")
            return

        # Tokenize training pairs
        def encode_pair(q, a):
            q_ids = self.tokenizer.encode(q)[:self.seq_len]
            a_ids = self.tokenizer.encode(a)[:self.seq_len]
            
            # Pad sequences
            q_ids += [self.pad_token_id] * (self.seq_len - len(q_ids))
            a_ids += [self.pad_token_id] * (self.seq_len - len(a_ids))
            
            return q_ids, a_ids

        encoded_pairs = [encode_pair(q, a) for q, a in train_pairs[:2000]]  # Increased for better training
        
        # Split into train/val
        train_data, val_data = train_test_split(encoded_pairs, test_size=0.1, random_state=42)

        # Create data loaders
        def create_dataset(data):
            queries = torch.tensor([q for q, _ in data], dtype=torch.long)
            answers = torch.tensor([a for _, a in data], dtype=torch.long)
            return torch.utils.data.TensorDataset(queries, answers)

        train_loader = torch.utils.data.DataLoader(
            create_dataset(train_data), batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            create_dataset(val_data), batch_size=batch_size
        )

        # Setup training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)

        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (queries, answers) in enumerate(train_loader):
                queries, answers = queries.to(self.device), answers.to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(queries, answers)
                loss = criterion(logits.view(-1, logits.size(-1)), answers.view(-1))
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'vocab_size': self.vocab_size
        }, save_path)
        
        logger.info(f"✅ Model saved to {save_path}")
        self.is_trained = True

    def generate_response(self, prompt, task="answer", max_length=100, temperature=0.8):
        if not self.is_trained:
            logger.warning("Model not trained, returning default response")
            return "Please train the model by uploading documents and calling train_on_documents."

        if not self.tokenizer.trained:
            logger.warning("Tokenizer not trained, returning default response")
            return "Please upload documents to train the tokenizer."

        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            return "Could not encode the prompt. Please check the input."

        prompt_ids = prompt_ids[:self.seq_len]
        prompt_ids += [self.pad_token_id] * (self.seq_len - len(prompt_ids))
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        context_embeddings = None
        relevant_info = ""
        if self.doc_texts:
            context_result = self._faq_chunk_search(prompt)
            if context_result and context_result.get("confidence", 0) > 0.1:
                relevant_info = context_result["summary"]
                context_ids = self.tokenizer.encode(relevant_info)
                context_ids = context_ids[:self.seq_len]
                context_ids += [self.pad_token_id] * (self.seq_len - len(context_ids))
                context_tensor = torch.tensor([context_ids], dtype=torch.long, device=self.device)

                with torch.no_grad():
                    context_embeddings = self.model.encode_document(context_tensor)
                    context_embeddings = context_embeddings.unsqueeze(1)

        with torch.no_grad():
            generated_ids = self.model.generate_with_context(
                prompt_tensor, 
                context_embeddings=context_embeddings,
                max_length=max_length,
                temperature=temperature,
                top_k=50
            )

            if generated_ids.numel() > 0:
                generated_text = self.tokenizer.decode(generated_ids[0].tolist())
                generated_text = self._post_process_response(generated_text)

                response = f"Generated Response: {generated_text}"
                if relevant_info:
                    response += f"\n\nContext Used: {relevant_info[:200]}..."
                    response += f"\nConfidence: {context_result['confidence']:.2f}"
                    response += f"\nSource: {context_result['source']}"
                return response
            else:
                return "Model failed to generate a response. Please try a different prompt."
        
    def _post_process_response(self, text):
        """Post-process generated text to improve quality"""
        # Remove special tokens and clean up
        text = text.replace('<UNK>', '').replace('<PAD>', '').replace('<SOS>', '').replace('<EOS>', '')
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        
        # Add period if missing
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text

    def _faq_chunk_search(self, prompt: str, top_k: int = 3):
        """Search for relevant chunks in documents"""
        if not self.doc_texts:
            return None
            
        all_chunks = []
        chunk_sources = []

        for fname, doc in self.doc_texts.items():
            sentences = doc.get("sentences", [])
            for sentence in sentences:
                if len(sentence.strip()) > 40:
                    all_chunks.append(sentence)
                    chunk_sources.append(fname)

        if not all_chunks:
            return None

        # Simple similarity search using embeddings
        prompt_embedding = get_embedding(prompt)
        chunk_embeddings = [get_embedding(chunk) for chunk in all_chunks]
        
        # Calculate similarities
        similarities = []
        for chunk_emb in chunk_embeddings:
            try:
                # Ensure embeddings have same dimension
                if len(prompt_embedding) != len(chunk_emb):
                    # Pad shorter embedding with zeros
                    max_len = max(len(prompt_embedding), len(chunk_emb))
                    prompt_embedding = np.pad(prompt_embedding, (0, max_len - len(prompt_embedding)))
                    chunk_emb = np.pad(chunk_emb, (0, max_len - len(chunk_emb)))
                
                sim = cosine_similarity([prompt_embedding], [chunk_emb])[0][0]
                similarities.append(sim)
            except:
                similarities.append(0.0)

        # Get top results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_chunks = [all_chunks[i] for i in top_indices if similarities[i] > 0.1]
        
        if not top_chunks:
            return None

        # Combine top chunks for summary
        combined_text = " ".join(top_chunks)
        avg_confidence = np.mean([similarities[i] for i in top_indices if similarities[i] > 0.1])
        sources = list(set(chunk_sources[i] for i in top_indices if similarities[i] > 0.1))

        return {
            "summary": combined_text[:500] + "..." if len(combined_text) >500 else combined_text,
            "confidence": avg_confidence,
            "source": ", ".join(sources)
        }

    def is_retriever_ready(self) -> bool:
        """Check if retriever is ready"""
        return bool(self.doc_texts)

    def get_status(self):
        """Get current status"""
        return {
            "trained": self.is_trained,
            "device": str(self.device),
            "documents": len(self.doc_texts),
            "tokenizer_vocab": len(self.tokenizer.vocab) if self.tokenizer.trained else 0,
            "embeddings": len(self.doc_embeddings)
        }

    def generate_with_beam_search(self, prompt, beam_size=3, max_length=100):
        """Generate response using beam search for better quality"""
        if not self.is_trained or not self.tokenizer.trained:
            return "❌ Model or tokenizer not trained yet."
        
        try:
            # Encode the prompt
            prompt_ids = self.tokenizer.encode(prompt)[:self.seq_len]
            prompt_ids += [self.pad_token_id] * (self.seq_len - len(prompt_ids))
            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
            
            # Get context if available
            context_embeddings = None
            if self.doc_texts:
                context_result = self._faq_chunk_search(prompt)
                if context_result and context_result.get("confidence", 0) > 0.1:
                    context_ids = self.tokenizer.encode(context_result["summary"])[:self.seq_len]
                    context_ids += [self.pad_token_id] * (self.seq_len - len(context_ids))
                    context_tensor = torch.tensor([context_ids], dtype=torch.long, device=self.device)
                    
                    with torch.no_grad():
                        context_embeddings = self.model.encode_document(context_tensor).unsqueeze(1)
            
            # Beam search implementation
            with torch.no_grad():
                self.model.eval()
                
                # Initialize beam
                beams = [([2], 0.0)]  # Start with SOS token and 0 log probability
                
                for _ in range(max_length):
                    candidates = []
                    
                    for sequence, score in beams:
                        if sequence[-1] == 3:  # EOS token
                            candidates.append((sequence, score))
                            continue
                        
                        # Get current input
                        current_input = torch.tensor([[sequence[-1]]], dtype=torch.long, device=self.device)
                        embedded = self.model.embedding(current_input)
                        
                        # Get encoder states for the sequence
                        if len(sequence) == 1:
                            embedded_input = self.model.embedding(prompt_tensor)
                            _, (hidden, cell) = self.model.encoder(embedded_input)
                            hidden = self.model._merge_bidir(hidden)
                            cell = self.model._merge_bidir(cell)
                        
                        # Generate next token probabilities
                        output, _ = self.model.decoder(embedded, (hidden, cell))
                        logits = self.model.output_layer(output)
                        probs = torch.nn.functional.log_softmax(logits.squeeze(), dim=-1)
                        
                        # Get top beam_size candidates
                        top_probs, top_indices = torch.topk(probs, beam_size)
                        
                        for i in range(beam_size):
                            new_sequence = sequence + [top_indices[i].item()]
                            new_score = score + top_probs[i].item()
                            candidates.append((new_sequence, new_score))
                    
                    # Select top beam_size candidates
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    beams = candidates[:beam_size]
                    
                    # Check if all beams ended
                    if all(seq[-1] == 3 for seq, _ in beams):
                        break
                
                # Get best sequence
                best_sequence = beams[0][0][1:]  # Remove SOS token
                if best_sequence and best_sequence[-1] == 3:
                    best_sequence = best_sequence[:-1]  # Remove EOS token
                
                # Decode and return
                if best_sequence:
                    generated_text = self.tokenizer.decode(best_sequence)
                    return self._post_process_response(generated_text)
                else:
                    return "❌ No valid sequence generated during beam search."

        except Exception as e:
            logger.error(f"Error in beam search generation: {e}")
            return f"❌ Error generating response with beam search: {str(e)}"

    def save_model(self, save_path="checkpoint.pt"):
        """Save the model and tokenizer"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'vocab_size': self.vocab_size
            }, save_path)
            self.tokenizer.save(self.tokenizer_path)
            logger.info(f"✅ Model and tokenizer saved successfully to {save_path} and {self.tokenizer_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return f"❌ Error saving model: {str(e)}"

    def load_model(self, model_path="checkpoint.pt", tokenizer_path="bpe_tokenizer.pkl"):
        """Load the model and tokenizer"""
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self._load_model()
        self._load_tokenizer()
        logger.info("✅ Model and tokenizer loaded successfully")

    def clear_documents(self):
        """Clear indexed documents"""
        self.doc_texts = {}
        self.doc_embeddings = {}
        logger.info("✅ Documents cleared")

    def get_document_summary(self, filename: str) -> str:
        """Get summary of a specific document"""
        if filename not in self.doc_texts:
            return f"❌ Document {filename} not found"
        
        doc = self.doc_texts[filename]
        sentences = doc.get("sentences", [])
        if not sentences:
            return "❌ No content available for summary"
        
        # Simple summary: take first few sentences
        summary_length = min(3, len(sentences))
        summary = " ".join(sentences[:summary_length])
        return self._post_process_response(summary[:500] + "..." if len(summary) > 500 else summary)
    def get_status(self):
        return {
            "trained": self.is_trained, # ← This must be True
            "device": str(self.device),
            "tokenizer_vocab": len(self.tokenizer.vocab) if self.tokenizer.trained else 0,
            "documents": len(self.doc_texts),
            "embeddings": len(self.doc_embeddings)
        }
    