# intent/intent_router.py
import torch
import json
from transformers import BertTokenizer, BertForSequenceClassification
from intent.classifier import BiLSTMClassifier
from intent.train_classifier import tokenize
import torch.nn.functional as F

class IntentRouter:
    def __init__(self, model_type="bert"):
        self.model_type = model_type.lower()

        if self.model_type == "bert":
            self.tokenizer = BertTokenizer.from_pretrained("bert_model")
            self.model = BertForSequenceClassification.from_pretrained("bert_model")
        elif self.model_type == "bilstm":
            with open("intent/intent_vocab.json") as f:
                vocab = json.load(f)
                self.stoi = vocab["stoi"]
                self.intent_labels = {int(k): v for k, v in vocab["intent_labels"].items()}

            self.model = BiLSTMClassifier(input_dim=len(self.stoi), output_dim=len(self.intent_labels))
            self.model.load_state_dict(torch.load("intent/intent_model.pth"))
        else:
            raise ValueError("Unsupported model type. Choose 'bert' or 'bilstm'.")

        self.model.eval()

    def predict_intent(self, query: str):
        if self.model_type == "bert":
            inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
        else:
            tokens = [self.stoi.get(tok, self.stoi.get("<unk>", 0)) for tok in tokenize(query)]
            inputs = torch.tensor(tokens).unsqueeze(0)
            with torch.no_grad():
                logits = self.model(inputs)

        probs = F.softmax(logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_label].item()

        label = ( "file" if pred_label == 0 else "general") if self.model_type == "bert" else self.intent_labels[pred_label]

        return label, confidence
