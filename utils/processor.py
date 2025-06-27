import re
import math
from typing import List
import logging

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 600, mode: str = "sentence") -> List[str]:
    """
    Chunk text based on either sentences or word count.
    
    Args:
        text (str): The input document text.
        chunk_size (int): Target size of each chunk (in words or characters depending on mode).
        mode (str): 'sentence' for punctuation-aware chunking, 'word' for word-block chunking.

    Returns:
        List[str]: List of text chunks.
    """
    if not text or len(text.strip()) < 50:
        logger.warning("⚠️ Provided text too short to chunk.")
        return []

    text = re.sub(r'\s+', ' ', text.strip())  # Normalize whitespace

    if mode == "sentence":
        # Break into sentences using punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""

        for sent in sentences:
            if not sent or len(sent.strip()) < 20:
                continue
            if len(current_chunk) + len(sent) <= chunk_size:
                current_chunk += sent + " "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sent + " "
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

    elif mode == "word":
        # Break based on word count
        words = text.split()
        num_chunks = math.ceil(len(words) / chunk_size)
        chunks = [
            " ".join(words[i * chunk_size: (i + 1) * chunk_size])
            for i in range(num_chunks)
        ]
    else:
        logger.error(f"❌ Unsupported chunk mode: {mode}")
        return []

    # Final filtering: remove junk chunks
    final_chunks = [chunk for chunk in chunks if len(chunk.split()) >= 10]
    logger.info(f"✅ Chunked into {len(final_chunks)} segment(s) using mode='{mode}'.")
    return final_chunks
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
