import json
import os

FEEDBACK_LOG = "logs/retriever_feedback.json"

def load_weights():
    if os.path.exists(FEEDBACK_LOG):
        with open(FEEDBACK_LOG, "r") as f:
            logs = json.load(f)
        weights = {"bm25": 0.3, "tfidf": 0.3, "vector": 0.4}
        positive_feedbacks = [l for l in logs if l["user_feedback"] == "good"]
        if positive_feedbacks:
            scores = {"bm25": 0, "tfidf": 0, "vector": 0}
            for entry in positive_feedbacks:
                for k in scores:
                    scores[k] += entry["retriever_weights"][k]
            total = sum(scores.values())
            weights = {k: v / total for k, v in scores.items()}
        return weights
    return {"bm25": 0.33, "tfidf": 0.33, "vector": 0.34}
