import re
import nltk
import spacy
from typing import List

# Download required models
nltk.download('punkt')

# Load spaCy English model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def chunk_text(text: str,
               method: str = "paragraph",
               chunk_size: int = 300,
               overlap: int = 50,
               tokenizer=None,
               max_tokens: int = 256,
               stride: int = 64,
               max_words: int = 100,
               window_size: int = 100) -> List[str]:
    """
    Universal chunking interface.
    
    Args:
        text (str): Raw input text.
        method (str): Chunking method: "paragraph", "token", "sliding", "sentence_nltk", "sentence_spacy".
        chunk_size (int): For paragraph chunker - maximum characters.
        overlap (int): Character overlap for paragraph-based chunker.
        tokenizer (transformers.PreTrainedTokenizer): Required for "token" method.
        max_tokens (int): Max tokens per chunk for "token" method.
        stride (int): Token overlap in "token" and "sliding" methods.
        max_words (int): Max words per chunk for sentence-based chunkers.
        window_size (int): For sliding window - words per chunk.
    
    Returns:
        List[str]: List of chunked text.
    """
    if method == "paragraph":
        return paragraph_chunker(text, chunk_size=chunk_size, overlap=overlap)
    elif method == "token":
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided for token-based chunking.")
        return token_chunker(text, tokenizer, max_tokens=max_tokens, stride=stride)
    elif method == "sliding":
        return sliding_window_chunker(text, window_size=window_size, stride=stride)
    elif method == "sentence_nltk":
        return sentence_chunker_nltk(text, max_words=max_words)
    elif method == "sentence_spacy":
        return sentence_chunker_spacy(text, max_words=max_words)
    else:
        raise ValueError(f"Unsupported chunking method: {method}")


# === Paragraph Chunking ===
def paragraph_chunker(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 1 <= chunk_size:
            current += " " + para if current else para
        else:
            chunks.append(current.strip())
            current = para

    if current:
        chunks.append(current.strip())

    final_chunks = []
    for i in range(len(chunks)):
        if i == 0:
            final_chunks.append(chunks[i])
        else:
            overlap_chunk = chunks[i - 1][-overlap:] if overlap > 0 else ""
            final_chunks.append((overlap_chunk + " " + chunks[i]).strip())

    return final_chunks


# === Token-Based Chunking ===
def token_chunker(text: str, tokenizer, max_tokens: int = 256, stride: int = 64) -> List[str]:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    for i in range(0, len(tokens), max_tokens - stride):
        chunk = tokens[i:i + max_tokens]
        decoded = tokenizer.decode(chunk)
        chunks.append(decoded.strip())

    return chunks


# === Sentence-Based Chunking (nltk) ===
def sentence_chunker_nltk(text: str, max_words: int = 100) -> List[str]:
    from nltk.tokenize import sent_tokenize, word_tokenize

    sentences = sent_tokenize(text)
    chunks, current = [], []
    word_count = 0

    for sentence in sentences:
        words = word_tokenize(sentence)
        if word_count + len(words) <= max_words:
            current.append(sentence)
            word_count += len(words)
        else:
            chunks.append(" ".join(current))
            current = [sentence]
            word_count = len(words)

    if current:
        chunks.append(" ".join(current))

    return chunks


# === Sentence-Based Chunking (spaCy) ===
def sentence_chunker_spacy(text: str, max_words: int = 100) -> List[str]:
    doc = nlp(text)
    chunks, current = [], []
    word_count = 0

    for sent in doc.sents:
        words = [token.text for token in sent if not token.is_space]
        if word_count + len(words) <= max_words:
            current.append(sent.text)
            word_count += len(words)
        else:
            chunks.append(" ".join(current))
            current = [sent.text]
            word_count = len(words)

    if current:
        chunks.append(" ".join(current))

    return chunks


# === Sliding Window Chunking ===
def sliding_window_chunker(text: str, window_size: int = 100, stride: int = 50) -> List[str]:
    words = text.split()
    chunks = []

    for i in range(0, len(words), stride):
        chunk = words[i:i + window_size]
        if chunk:
            chunks.append(" ".join(chunk))

    return chunks
