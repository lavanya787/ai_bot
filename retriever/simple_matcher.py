from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def simple_matcher(query, text, threshold=0.4):
    chunks = [para for para in text.split("\n") if len(para.strip()) > 0]
    if not chunks:
        return "No content found in file.", "Error"

    vectorizer = TfidfVectorizer().fit([query] + chunks)
    vectors = vectorizer.transform([query] + chunks)
    sims = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    top_idx = sims.argmax()
    best_match = chunks[top_idx]
    score = sims[top_idx]

    if score > threshold:
        return best_match, "Matched"
    else:
        return "No relevant answer found.", "Not Found"
