# models/general_search_model.py
import torch
import pickle
from models.qa_model import SimpleQAModel
from utils.search_google import search_google

def tokenize(text):
    return text.lower().split()

def encode(text, vocab, max_len=20):
    tokens = tokenize(text)
    token_ids = [vocab.get(t, vocab['<unk>']) for t in tokens]
    return token_ids[:max_len] + [0] * (max_len - len(token_ids))

class GeneralSearchQA:
    def __init__(self, model_path, vocab_path):
        self.vocab = pickle.load(open(vocab_path, 'rb'))
        self.model = SimpleQAModel(len(self.vocab))
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def answer(self, question):
        context = search_google(question)
        candidates = context.split('\n')  # break into sentences

        best_score = float('-inf')
        best_sent = "No suitable answer found."

        for sent in candidates:
            tokens = torch.tensor([encode(sent, self.vocab)])
            with torch.no_grad():
                score = self.model(tokens).item()
                if score > best_score:
                    best_score = score
                    best_sent = sent
        return best_sent
