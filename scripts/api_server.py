# scripts/api_server.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
from models.transformer_generator import TransformerGenerator
from tokenizer.tokenizer import CustomTokenizer
import uvicorn

app = FastAPI()

# Load model and tokenizer
model = TransformerGenerator.load_from_checkpoint("saved_models/general/model.pt")
tokenizer = CustomTokenizer.load("tokenizer/vocab.json")

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    mode: str = "greedy"  # Options: greedy, top_k, top_p, beam

@app.post("/generate/")
def generate_text(req: PromptRequest):
    tokens = tokenizer.encode(req.prompt)
    output_ids = model.generate(
        input_ids=tokens,
        max_length=req.max_tokens,
        mode=req.mode
    )
    generated = tokenizer.decode(output_ids)
    return {"prompt": req.prompt, "response": generated}

if __name__ == "__main__":
    uvicorn.run("scripts.api_server:app", host="0.0.0.0", port=8000, reload=True)
