import pandas as pd
from retriever.hybrid_retriever import retrieve_context

def benchmark_retrievers(csv_path, top_k=5, return_metrics=False):
    df = pd.read_csv(csv_path)
    total, hit = 0, 0

    for _, row in df.iterrows():
        query = row["question"]
        answer = row["answer"]
        context = retrieve_context(query, k=top_k)
        context_str = " ".join(context).lower()
        if answer.lower() in context_str:
            hit += 1
        total += 1

    acc = hit / total
    if return_metrics:
        return acc, total, hit
    else:
        print(f"✅ Retriever Accuracy@{top_k}: {hit}/{total} = {acc:.2%}")
