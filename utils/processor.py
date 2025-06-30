import re
import math
from typing import List
import logging

logger = logging.getLogger(__name__)


# file_processing/processor.py
import re

def chunk_text(docs, max_chunk_len=200, overlap=40):
    """
    Splits documents into token-like chunks with optional overlap.
    Uses sentence boundaries when possible.

    :param docs: list of strings (corpus)
    :param max_chunk_len: max tokens per chunk (approximate)
    :param overlap: overlap between chunks to preserve context
    :return: list of text chunks
    """
    chunks = []
    for doc in docs:
        doc = re.sub(r"\s+", " ", doc.strip())
        sentences = re.split(r'(?<=[.?!])\s+', doc)

        current_chunk = []
        current_len = 0

        for sentence in sentences:
            sent_len = len(sentence.split())

            if current_len + sent_len <= max_chunk_len:
                current_chunk.append(sentence)
                current_len += sent_len
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                # start next chunk with optional overlap
                overlap_tokens = " ".join(current_chunk[-overlap:]) if overlap and current_chunk else ""
                current_chunk = [overlap_tokens, sentence] if overlap_tokens else [sentence]
                current_len = len(" ".join(current_chunk).split())

        if current_chunk:
            chunks.append(" ".join(current_chunk))
    return chunks

def get_chunking_config(filename: str) -> dict:
    """
    Dynamically selects chunking mode and size based on file type or name.

    Returns:
        dict: {mode: 'sentence' or 'word', chunk_size: int}
    """
    name = filename.lower()

    if name.endswith(".csv") or name.endswith(".json"):
        return {"mode": "word", "chunk_size": 80}  # row-based structure
    elif name.endswith(".txt"):
        return {"mode": "sentence", "chunk_size": 500}
    elif name.endswith(".pdf"):
        return {"mode": "sentence", "chunk_size": 600}
    elif name.endswith(".docx"):
        return {"mode": "sentence", "chunk_size": 600}
    elif name.endswith((".png", ".jpg", ".jpeg")):
        return {"mode": "word", "chunk_size": 100}  # OCR text is unpredictable
    else:
        # Default fallback
        return {"mode": "sentence", "chunk_size": 1000}
