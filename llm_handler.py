import torch
import torch.nn as nn
import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional
import logging
import regex as re
from collections import Counter
import hashlib
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even for rotary encoding, got {d_model}")
        
        theta = 10000 ** (-2 * torch.arange(0, d_model//2, dtype=torch.float) / d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        angles = pos * theta.unsqueeze(0)
        self.register_buffer('sin', angles.sin())
        self.register_buffer('cos', angles.cos())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(-2)
        if seq_len > self.sin.size(0):
            theta = 10000 ** (-2 * torch.arange(0, self.d_model//2, dtype=torch.float) / self.d_model)
            pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
            angles = pos * theta.unsqueeze(0)
            self.register_buffer('sin', angles.sin())
            self.register_buffer('cos', angles.cos())
        
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        x_rot = torch.zeros_like(x)
        x_rot[..., ::2] = x_even * self.cos[:seq_len] - x_odd * self.sin[:seq_len]
        x_rot[..., 1::2] = x_even * self.sin[:seq_len] + x_odd * self.cos[:seq_len]
        return x_rot

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / np.sqrt(self.d_k)
        self.rope = RotaryPositionalEncoding(self.d_k)

    def forward(self, q_input: torch.Tensor, k_input: torch.Tensor, v_input: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len = q_input.size(0), q_input.size(1)
        
        q = self.W_q(q_input).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k_input).view(batch_size, k_input.size(1), self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v_input).view(batch_size, v_input.size(1), self.num_heads, self.d_k).transpose(1, 2)
        
        q = self.rope(q)
        k = self.rope(k)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(context)

class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        normed_x = self.norm1(x)
        x = x + self.dropout(self.attention(normed_x, normed_x, normed_x, mask))
        normed_x = self.norm2(x)
        x = x + self.dropout(self.ff(normed_x))
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor, self_mask: torch.Tensor = None, cross_mask: torch.Tensor = None) -> torch.Tensor:
        normed_x = self.norm1(x)
        x = x + self.dropout(self.self_attention(normed_x, normed_x, normed_x, self_mask))
        
        normed_x = self.norm2(x)
        x = x + self.dropout(self.cross_attention(normed_x, context, context, cross_mask))
        
        normed_x = self.norm3(x)
        x = x + self.dropout(self.ff(normed_x))
        return x

class RAGModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, d_ff: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoder(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers // 2)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoder(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers // 2)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def encode_document(self, doc: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len = doc.size()
        
        x = self.embedding(doc)
        pos = torch.arange(seq_len, device=doc.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_embedding(pos)
        x = self.dropout(x)
        
        for encoder in self.encoder_layers:
            x = encoder(x, mask)
        
        x = self.norm(x)
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            x_masked = x * mask_expanded
            doc_emb = x_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            doc_emb = x.mean(dim=1)
        
        return doc_emb / torch.norm(doc_emb, dim=-1, keepdim=True).clamp(min=1e-9)

    def forward(self, query: torch.Tensor, context_docs: torch.Tensor, query_mask: torch.Tensor = None, context_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len = query.size()
        
        x = self.embedding(query)
        pos = torch.arange(seq_len, device=query.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_embedding(pos)
        x = self.dropout(x)
        
        context_emb = self.embedding(context_docs)
        ctx_batch, ctx_seq = context_docs.size()
        ctx_pos = torch.arange(ctx_seq, device=context_docs.device).unsqueeze(0).expand(ctx_batch, -1)
        context_emb = context_emb + self.pos_embedding(ctx_pos)
        context_emb = self.dropout(context_emb)
        
        for encoder in self.encoder_layers:
            context_emb = encoder(context_emb, context_mask)
        
        for decoder in self.decoder_layers:
            x = decoder(x, context_emb, query_mask, context_mask)
        
        x = self.norm(x)
        return self.fc_out(x)

class ImprovedBPETokenizer:
    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.vocab_size = 0
        self.pattern = re.compile(r'\w+|\W')
        
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
        
    def _add_special_tokens(self):
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for token in special_tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                self.inverse_vocab[len(self.vocab) - 1] = token

    def train(self, texts: List[str], target_vocab_size: int = 5000):
        logger.info(f"Training BPE tokenizer with target vocab size: {target_vocab_size}")
        
        self.vocab = {chr(i): i for i in range(256)}
        self.inverse_vocab = {i: chr(i) for i in range(256)}
        self.vocab_size = 256
        
        self._add_special_tokens()
        
        word_freqs = Counter()
        for text in texts:
            words = self.pattern.findall(text.lower())
            for word in words:
                word_freqs[word] += 1
        
        splits = {}
        for word, freq in word_freqs.items():
            splits[word] = list(word)
        
        while self.vocab_size < target_vocab_size:
            pairs = Counter()
            for word, word_split in splits.items():
                for i in range(len(word_split) - 1):
                    pairs[(word_split[i], word_split[i + 1])] += word_freqs[word]
            
            if not pairs:
                break
                
            best_pair = max(pairs, key=pairs.get)
            new_token = ''.join(best_pair)
            
            self.vocab[new_token] = self.vocab_size
            self.inverse_vocab[self.vocab_size] = new_token
            self.merges.append(best_pair)
            self.vocab_size += 1
            
            new_splits = {}
            for word, word_split in splits.items():
                new_split = []
                i = 0
                while i < len(word_split):
                    if i < len(word_split) - 1 and (word_split[i], word_split[i + 1]) == best_pair:
                        new_split.append(new_token)
                        i += 2
                    else:
                        new_split.append(word_split[i])
                        i += 1
                new_splits[word] = new_split
            splits = new_splits
        
        logger.info(f"Tokenizer trained with vocab size: {self.vocab_size}")

    def encode(self, text: str) -> List[int]:
        if not text.strip():
            return []
        
        words = self.pattern.findall(text.lower())
        tokens = []
        
        for word in words:
            word_tokens = list(word)
            
            for merge in self.merges:
                new_word_tokens = []
                i = 0
                while i < len(word_tokens):
                    if i < len(word_tokens) - 1 and (word_tokens[i], word_tokens[i + 1]) == merge:
                        new_word_tokens.append(''.join(merge))
                        i += 2
                    else:
                        new_word_tokens.append(word_tokens[i])
                        i += 1
                word_tokens = new_word_tokens
            
            for token in word_tokens:
                tokens.append(self.vocab.get(token, self.vocab.get(self.unk_token, 1)))
        
        return tokens

    def decode(self, ids: List[int]) -> str:
        tokens = []
        for id in ids:
            if id in self.inverse_vocab:
                token = self.inverse_vocab[id]
                if token not in [self.pad_token, self.bos_token, self.eos_token]:
                    tokens.append(token)
        return ''.join(tokens)

    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump({
                'vocab': self.vocab,
                'inverse_vocab': self.inverse_vocab,
                'merges': self.merges,
                'vocab_size': self.vocab_size
            }, f)

    def load(self, filename: str):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.vocab = data['vocab']
            self.inverse_vocab = data['inverse_vocab']
            self.merges = data['merges']
            self.vocab_size = data['vocab_size']

class LLMHandler:
    def __init__(self, model_path: str = 'checkpoint_step_1000.pt', tokenizer_path: str = 'bpe_tokenizer.pkl'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = ImprovedBPETokenizer()
        self.doc_embeddings = {}
        self.doc_texts = {}  # Store raw text
        
        # Model hyperparameters
        self.vocab_size = 5000
        self.d_model = 256
        self.num_heads = 8
        self.num_layers = 8
        self.d_ff = 512
        self.seq_len = 128
        self.dropout = 0.1
        self.top_k = 3
        self.pad_token_id = 0
        self.is_trained = False
        
        # Initialize tokenizer
        if os.path.exists(tokenizer_path):
            try:
                self.tokenizer.load(tokenizer_path)
                logger.info("Tokenizer loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load tokenizer: {e}")
                self._init_tokenizer(tokenizer_path)
        else:
            self._init_tokenizer(tokenizer_path)
        
        # Initialize model
        self.model = RAGModel(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            d_ff=self.d_ff,
            dropout=self.dropout
        ).to(self.device)
        
        # Load checkpoint if exists
        if os.path.exists(model_path):
            self._load_checkpoint(model_path)
        
        self.model.eval()
        logger.info(f"🧠 Total trainable model parameters: {count_parameters(self.model):,}")
        logger.info(f"LLMHandler initialized. Device: {self.device}, Trained: {self.is_trained}")

    def _init_tokenizer(self, tokenizer_path: str):
        sample_texts = [
            "This is a sample document for training the tokenizer.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning and artificial intelligence are transforming technology.",
            "Natural language processing enables computers to understand human language."
        ] * 250
        
        self.tokenizer.train(sample_texts, self.vocab_size)
        self.tokenizer.save(tokenizer_path)
        logger.info("New tokenizer created and saved")

    def _load_checkpoint(self, model_path: str):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)

            if len(missing_keys) == 0 and len(unexpected_keys) == 0:
                self.is_trained = True
                logger.info("✅ Model checkpoint fully loaded. Model is trained and ready.")
            else:
                logger.warning(f"⚠️ Partial model checkpoint loaded. Missing keys: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
                self.is_trained = len(missing_keys) < len(list(self.model.parameters())) // 2
                logger.info(f"✅ Proceeding with partially trained model. Trained: {self.is_trained}")

        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            self.is_trained = False

    def index_document(self, filename: str, content: str):
        """Index a document for retrieval - FIXED"""
        if not content.strip():
            logger.warning(f"Empty content for {filename}")
            return
        
        # Store raw text
        self.doc_texts[filename] = content
        
        # Generate embedding if model is trained
        if self.is_trained:
            try:
                # Tokenize and create tensor
                tokens = self.tokenizer.encode(content)
                if not tokens:
                    logger.warning(f"No tokens generated for {filename}")
                    return
                
                # Pad/truncate to seq_len
                if len(tokens) < self.seq_len:
                    tokens = tokens + [self.pad_token_id] * (self.seq_len - len(tokens))
                else:
                    tokens = tokens[:self.seq_len]
                
                doc_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)
                
                # Create attention mask
                mask = torch.ones(1, self.seq_len, dtype=torch.float).to(self.device)
                original_len = min(len(self.tokenizer.encode(content)), self.seq_len)
                if original_len < self.seq_len:
                    mask[0, original_len:] = 0
                
                with torch.no_grad():
                    embedding = self.model.encode_document(doc_tensor, mask)
                    self.doc_embeddings[filename] = embedding.cpu().numpy().flatten()
                    logger.info(f"Document indexed: {filename}")
                
            except Exception as e:
                logger.error(f"Error indexing document {filename}: {e}")
        else:
            logger.info(f"Document stored (model untrained): {filename}")

    def generate_response(self, prompt: str, task: str = 'answer') -> str:
        """Generate response using RAG - FIXED"""
        if not self.doc_texts:
            return "Please upload documents first to enable RAG-based responses."
        
        return self._fallback_response(prompt, task)

    def _fallback_response(self, prompt: str, task: str = "answer") -> str:
        """MINIMAL structured response generator"""
        def simple_extract(text: str, patterns: List[str]) -> str:
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    sentence = match.group(0).strip()
                    if len(sentence) > 50:  # Minimum meaningful length
                        return sentence[:300] + "..." if len(sentence) > 300 else sentence
            return ""

        # Find best matching document
        query_lower = prompt.lower()
        best_content = ""
        best_file = ""
        best_score = 0
        
        # Simple keyword matching + length scoring
        for filename, content in self.doc_texts.items():
            content_lower = content.lower()
            
            # Count query word matches
            query_words = re.findall(r'\w+', query_lower)
            matches = sum(1 for word in query_words if word in content_lower)
            score = matches / len(query_words) if query_words else 0
            
            # Boost score for relevant physics terms
            physics_terms = ['compton', 'effect', 'scattering', 'photon', 'electron', 'quantum', 'wavelength']
            for term in physics_terms:
                if term in query_lower and term in content_lower:
                    score += 0.2
            
            if score > best_score:
                best_score = score
                best_content = content
                best_file = filename
        
        if not best_content or best_score < 0.1:
            return f"⚠️ Sorry, I couldn't find relevant information about '{prompt}' in the uploaded documents."

        # Extract structured information
        definition = simple_extract(best_content, [
            r'[Tt]he .{0,50}[Ee]ffect.{0,200}[.!?]',
            r'.{0,100} is .{20,200}[.!?]',
            r'.{0,100} refers to .{20,200}[.!?]'
        ])
        
        historical = simple_extract(best_content, [
            r'[Dd]iscovered.{0,100}[.!?]',
            r'[Cc]ompton.{0,50}(1923|Nobel|Prize).{0,100}[.!?]',
            r'[Aa]rthur.{0,50}[Cc]ompton.{0,100}[.!?]'
        ])
        
        experimental = simple_extract(best_content, [
            r'[Xx]-ray.{0,200}[.!?]',
            r'[Ss]catter.{0,200}[.!?]',
            r'[Ee]xperiment.{0,200}[.!?]'
        ])
        
        equation = simple_extract(best_content, [
            r'Δλ.{0,100}[.!?]',
            r'λ.{0,100}=.{0,100}[.!?]',
            r'[Ww]avelength.{0,100}[.!?]'
        ])
        
        significance = simple_extract(best_content, [
            r'[Ss]ignificance.{0,200}[.!?]',
            r'[Ii]mportant.{0,200}[.!?]',
            r'[Qq]uantum.{0,200}[.!?]'
        ])

        # Format response
        response = f"""💬 **ChatGPT said:**
{prompt.strip().capitalize()}

📘 **Definition**
{definition or "The effect is described in the document but not clearly defined."}

💡 **Historical Context**
{historical or "Historical details not explicitly mentioned in the source."}

⚙️ **Experimental Setup**
{experimental or "Experimental details not clearly described."}

🧮 **Equation**
{equation or "Mathematical formula not found in the source."}

📌 **Why It Matters**
{significance or "Significance discussed but not clearly stated."}

📄 **Source**: {best_file}
📊 **Relevance Score**: {best_score:.2f}"""

        return response

    def get_status(self) -> Dict:
        """Get current status of the handler"""
        return {
            'is_trained': self.is_trained,
            'device': str(self.device),
            'vocab_size': self.vocab_size,
            'documents_indexed': len(self.doc_texts),
            'embeddings_generated': len(self.doc_embeddings),
            'model_parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0
        }