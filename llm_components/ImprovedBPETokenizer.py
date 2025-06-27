# llm_components/ImprovedBPETokenizer.py

import re
import pickle
from collections import defaultdict, Counter

class ImprovedBPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.word_to_token = {}
        self.token_to_word = {}
        self.word_freq = {}

    def train(self, corpus, vocab_size=5000):
        words = re.findall(r'\w+', ' '.join(corpus).lower())
        self.word_freq = Counter(words)
        self.vocab = {}
        for i, word in enumerate(self.word_freq):
            self.vocab[word] = i
            self.word_to_token[word] = i
            self.token_to_word[i] = word
        self._truncate_vocab(vocab_size)

    def encode(self, text):
        tokens = []
        words = re.findall(r'\w+', text.lower())
        for word in words:
            token = self.word_to_token.get(word)
            if token is not None:
                tokens.append(token)
        return tokens

    def decode(self, tokens):
        return ' '.join([self.token_to_word.get(t, '<UNK>') for t in tokens])

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'vocab': self.vocab,
                'word_to_token': self.word_to_token,
                'token_to_word': self.token_to_word,
                'word_freq': self.word_freq
            }, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            try:
                self.vocab = data['vocab']
                self.word_to_token = data['word_to_token']
                self.token_to_word = data['token_to_word']
                self.word_freq = data['word_freq']
            except KeyError as e:
                raise ValueError(f"Tokenizer file missing key: {e}. Try deleting and retraining.")
    
    def _truncate_vocab(self, limit):
        # Keep only top `limit` frequent words
        sorted_vocab = sorted(self.word_freq.items(), key=lambda x: -x[1])[:limit]
        self.vocab = {}
        self.word_to_token = {}
        self.token_to_word = {}
        for i, (word, _) in enumerate(sorted_vocab):
            self.vocab[word] = i
            self.word_to_token[word] = i
            self.token_to_word[i] = word
