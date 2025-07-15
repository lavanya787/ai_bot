import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import re

# Define constants for file paths
EMBEDDING_PATH = "embedding_store/embeddings.npy"
CHUNK_PATH = "embedding_store/chunks.pkl"

# Custom Transformer Model
class CustomTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=2):
        super(CustomTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True  
        )
        self.fc = nn.Linear(d_model, d_model)
        self.d_model = d_model

    def forward(self, input_ids):
        embedded = self.embedding(input_ids) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        output = self.transformer(embedded, embedded)
        output = self.fc(output)
        return output

# Custom Tokenizer
class SimpleTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(vocab)}
        self.pad_token_id = self.word2idx.get("[PAD]", 0)
        self.unk_token_id = self.word2idx.get("[UNK]", 1)

    def encode(self, text, max_length=512):
        words = re.findall(r'\w+|[^\w\s]', text.lower(), re.UNICODE)
        tokens = [self.word2idx.get(word, self.unk_token_id) for word in words][:max_length]
        tokens += [self.pad_token_id] * (max_length - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

    def batch_encode(self, texts, max_length=512):
        return torch.stack([self.encode(text, max_length) for text in texts])

# QAHandler Class
class QAHandler:
    def __init__(self):
        self.vocab = load_vocab()
        self.model = load_transformer(len(self.vocab))
        self.tokenizer = SimpleTokenizer(self.vocab)
        self.retriever = self._load_vector_db()

    def _load_vector_db(self):
        def is_valid(path):
            return os.path.exists(path) and os.path.getsize(path) > 0
        
        if not is_valid(EMBEDDING_PATH) or not is_valid(CHUNK_PATH):
            print("⚠️ Embedding or chunk file missing or invalid — building from fallback corpus...")
            embeddings, chunks = self._build_embeddings_and_chunks()
        else:
            embeddings = np.load(EMBEDDING_PATH).astype("float32")
            with open(CHUNK_PATH, "rb") as f:
                chunks = pickle.load(f)
        return embeddings, chunks

    def _build_embeddings_and_chunks(self):
        texts = []
        corpus_path = "corpus"
        
        # Create corpus directory and a default file if it doesn't exist
        if not os.path.exists(corpus_path):
            print(f"Creating corpus directory at '{corpus_path}'...")
            os.makedirs(corpus_path, exist_ok=True)
            # Create a default sample file
            default_file = os.path.join(corpus_path, "sample.txt")
            with open(default_file, "w", encoding="utf-8") as f:
                f.write(
                    "This is a default sample document for testing the QA system.\n"
                    "It contains multiple sentences to ensure chunking and embedding work correctly.\n"
                    "Add more text files to this corpus directory for better performance."
                )
        
        # Load texts from corpus
        for file in os.listdir(corpus_path):
            file_path = os.path.join(corpus_path, file)
            if os.path.isfile(file_path) and file.endswith('.txt'):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content and len(content) >= 10:
                            texts.append(content)
                except Exception as e:
                    print(f"⚠️ Error reading {file_path}: {e}")
        
        if not texts:
            print("⚠️ No valid texts found in corpus. Using default sample text.")
            texts = [
                "This is a fallback sample document for testing the QA system."
            ]

        chunks = []
        for text in texts:
            chunks.extend(chunk_text(text))
        
        if not chunks:
            raise ValueError("No valid chunks created from corpus. Check corpus content or chunk_text function.")

        embeddings = get_embeddings_batch(chunks, self.model, self.tokenizer)
        os.makedirs(os.path.dirname(EMBEDDING_PATH), exist_ok=True)
        np.save(EMBEDDING_PATH, np.array(embeddings))
        with open(CHUNK_PATH, "wb") as f:
            pickle.dump(chunks, f)
        return embeddings, chunks

# Helper Functions
def load_vocab():
    vocab = {"[PAD]", "[UNK]"}
    corpus_path = "corpus"
    if os.path.exists(corpus_path):
        for file in os.listdir(corpus_path):
            file_path = os.path.join(corpus_path, file)
            if os.path.isfile(file_path) and file.endswith('.txt'):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().lower()
                        words = re.findall(r'\w+|[^\w\s]', content, re.UNICODE)
                        vocab.update(words)
                except Exception as e:
                    print(f"⚠️ Error reading {file_path} for vocab: {e}")
    if len(vocab) < 2:
        vocab = {"[PAD]", "[UNK]"} | {f"word{i}" for i in range(10000)}
    return list(vocab)

def load_transformer(vocab_size):
    return CustomTransformer(vocab_size=vocab_size, d_model=512, nhead=8, num_layers=2)

def chunk_text(text, chunk_size=512):
    if not text or len(text.strip()) < 10:
        return []
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size) if len(" ".join(words[i:i + chunk_size]).strip()) >= 10]

def get_embeddings_batch(batch, model, tokenizer):
    if not batch:
        return np.array([])
    
    max_length = 512
    input_ids = tokenizer.batch_encode(batch, max_length=max_length)
    
    with torch.no_grad():
        outputs = model(input_ids)
    
    embeddings = outputs.mean(dim=1).cpu().numpy()
    return embeddings

# Initialize a global QAHandler for get_embedding
_qa_handler = QAHandler()

def get_embedding(text):
    """Get embedding for a single text."""
    if not text or len(text.strip()) < 10:
        return np.array([])
    return get_embeddings_batch([text], _qa_handler.model, _qa_handler.tokenizer)[0]