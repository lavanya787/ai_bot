# scripts/inference_cli.py

import torch
from models.transformer_generator import TransformerGenerator
from tokenizer.tokenizer import CustomTokenizer
from retriever.hybrid_retriever import retrieve_context
from utils.prompt_templates import apply_persona

# Load model and tokenizer
model = TransformerGenerator.load_from_checkpoint("saved_models/general/model.pt")
tokenizer = CustomTokenizer.load("tokenizer/vocab.json")
model.eval()

def cli():
    print("🧠 Custom AI Bot (Type 'exit' to quit)\n")
    mode = input("Use RAG? (yes/no): ").strip().lower() == "yes"
    style = input("Choose style (default, tech_support, child_friendly, shakespeare, sarcastic): ").strip()

    while True:
        prompt = input("\n👤 You: ")
        if prompt.lower() in ["exit", "quit"]:
            break

        # Build final prompt
        if mode:
            final_prompt = retrieve_context(prompt, style)
        else:
            final_prompt = apply_persona(prompt, style)

        input_ids = tokenizer.encode(final_prompt)
        input_tensor = torch.tensor(input_ids).unsqueeze(0)  # shape: [1, seq_len]

        with torch.no_grad():
            output_ids = model.generate(input_tensor, max_length=50, mode="greedy")

        generated = tokenizer.decode(output_ids[0])
        print(f"🤖 Bot: {generated}")

if __name__ == "__main__":
    cli()
