import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ✅ 1. Toy Tokenizer
def simple_tokenizer(text):
    return text.lower().split()

# ✅ 2. Vocabulary Builder
class Vocab:
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = ['<PAD>', '<UNK>']
    
    def build(self, texts):
        for text in texts:
            for word in simple_tokenizer(text):
                if word not in self.word2idx:
                    self.idx2word.append(word)
                    self.word2idx[word] = len(self.idx2word) - 1

    def encode(self, text, max_len=10):
        tokens = simple_tokenizer(text)
        ids = [self.word2idx.get(token, 1) for token in tokens]
        return ids[:max_len] + [0] * (max_len - len(ids))

# ✅ 3. Custom Dataset
class QADataset(Dataset):
    def __init__(self, questions, answers, vocab, max_len=10):
        self.questions = questions
        self.answers = answers
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        q_ids = self.vocab.encode(self.questions[idx], self.max_len)
        a_ids = self.vocab.encode(self.answers[idx], self.max_len)
        input_ids = torch.tensor(q_ids + a_ids)  # simple concat
        return input_ids

# ✅ 4. Custom Feedforward QA Model
class SimpleQAModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_classes=2):
        super(SimpleQAModel, self).__init__()

        # ✅ Embedding Layer (learned from scratch)
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # 🔷 Embeddings

        # ✅ Feedforward Layers
        self.fc1 = nn.Linear(embed_dim * 20, hidden_dim)       # 🔶 Feedforward NN
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)          # 🔶 Classifier Layer

    def forward(self, input_ids):
        x = self.embedding(input_ids)          # [batch_size, seq_len, embed_dim]
        x = x.view(x.size(0), -1)              # flatten
        x = self.relu(self.fc1(x))             # hidden layer
        logits = self.fc2(x)                   # output layer
        return logits

# ✅ 5. Data Prep
questions = ["What is AI?", "Define machine learning."]
answers = ["AI is artificial intelligence.", "ML is a subset of AI."]
vocab = Vocab()
vocab.build(questions + answers)

dataset = QADataset(questions, answers, vocab)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# ✅ 6. Model Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleQAModel(vocab_size=len(vocab.idx2word)).to(device)

# ✅ 7. Loss & Optimizer
criterion = nn.CrossEntropyLoss()      # 🎯 Loss Function
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # 🔁 Backpropagation engine

# ✅ 8. Dummy labels
labels = torch.tensor([0, 1]).long().to(device)  # pretend each sample has a unique label

# ✅ 9. Training Loop
for epoch in range(5):
    for i, batch in enumerate(loader):
        input_ids = batch.to(device)

        # 🔁 Feedforward
        logits = model(input_ids)

        # 🔁 Compute Loss
        loss = criterion(logits, labels[:logits.size(0)])

        # 🔁 Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

# ✅ 10. Save the model
torch.save(model.state_dict(), "saved_models/general/qa_model.pkl")
print("✅ Model saved without pre-trained dependency")

# 🧠 RLHF (Reinforcement Learning with Human Feedback) would come here
#       Usually done after pretraining + supervised fine-tuning
#       Requires human-labeled rewards and PPO algorithm (see: training/ppo_trainer.py)

# 🌀 Sampling Algorithms (greedy, top-k) apply during **inference/generation**, not training
