# training/reward_dataset.py

import json
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import pandas as pd

def load_feedback(path="logs/rlhf_feedback.json"):
    with open(path, "r") as f:
        return json.load(f)

def create_preference_pairs(feedback_data):
    # Group by prompt
    grouped = defaultdict(lambda: {"👍": [], "👎": []})
    for item in feedback_data:
        grouped[item["prompt"]][item["feedback"]].append(item["response"])

    # Build pairs: (prompt, preferred, rejected)
    preference_pairs = []
    for prompt, fb in grouped.items():
        pos = fb["👍"]
        neg = fb["👎"]
        min_len = min(len(pos), len(neg))
        if min_len == 0:
            continue  # Skip if no valid pairing

        for i in range(min_len):
            preference_pairs.append({
                "prompt": prompt,
                "preferred": pos[i],
                "rejected": neg[i]
            })

    return preference_pairs

def save_pairs(pairs, out_path="training_data/rlhf_pairs.json"):
    with open(out_path, "w") as f:
        json.dump(pairs, f, indent=4)

def visualize_feedback(path="logs/rlhf_feedback.json"):
    with open(path, "r") as f:
        feedback = json.load(f)

    df = pd.DataFrame(feedback)
    if 'user' not in df.columns:
        df["user"] = "anonymous"

    counts = df.groupby(["user", "feedback"]).size().unstack().fillna(0)
    counts.plot(kind="bar", stacked=True)
    plt.title("Feedback Distribution by User")
    plt.ylabel("Count")
    plt.xlabel("User")
    plt.tight_layout()
    plt.savefig("logs/feedback_distribution.png")
    plt.show()

def should_trigger_training(path="logs/rlhf_feedback.json", threshold=20):
    with open(path, "r") as f:
        feedback = json.load(f)
    return len(feedback) >= threshold

if __name__ == "__main__":
    if should_trigger_training():
        feedback = load_feedback()
        pairs = create_preference_pairs(feedback)
        save_pairs(pairs)
        print("✅ Triggering PPO fine-tuning...")
        os.system("python scripts/ppo_trainer.py")
