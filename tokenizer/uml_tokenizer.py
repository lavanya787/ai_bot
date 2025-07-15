import math
import json
import pickle
import logging
from collections import defaultdict, Counter
from typing import List, Optional, Union

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnigramTokenizer:
    def __init__(
        self,
        vocab: Optional[dict] = None,
        vocab_path: Optional[str] = None,
        max_subword_length: int = 10,
        end_of_word_token: str = "</w>",
        vocab_size: int = 500
    ):
        self.vocab = vocab or {}
        self.max_subword_length = max_subword_length
        self.eow = end_of_word_token
        self.vocab_size = vocab_size
        self.trained = bool(self.vocab)
        self.token_to_id = {}
        self.id_to_token = {}
        self.token_freqs = Counter()

        if vocab_path:
            self.load(vocab_path)

        # Ensure <UNK> always exists
        if "<UNK>" not in self.token_to_id:
            self.token_to_id["<UNK>"] = len(self.token_to_id)
            self.id_to_token[self.token_to_id["<UNK>"]] = "<UNK>"

    def _all_substrings(self, word: str) -> List[str]:
        word += self.eow
        subs = []
        for i in range(len(word)):
            for j in range(i + 1, min(len(word), i + self.max_subword_length) + 1):
                subs.append(word[i:j])
        return subs

    def build_initial_vocab(self, corpus: List[str], vocab_size: int) -> dict:
        counter = Counter()
        for line in corpus:
            for word in line.strip().split():
                subs = self._all_substrings(word)
                counter.update(subs)

        total = sum(counter.values())
        return {
            sub: math.log(freq / total)
            for sub, freq in counter.most_common(vocab_size)
        }

    def _segmentations(self, word: str) -> List[List[str]]:
        word += self.eow
        results = []

        def split(pos: int, path: List[str]):
            if pos == len(word):
                results.append(path)
                return
            for end in range(pos + 1, min(len(word), pos + self.max_subword_length) + 1):
                sub = word[pos:end]
                if sub in self.vocab:
                    split(end, path + [sub])

        split(0, [])
        return results

    def _best_segmentation(self, word: str) -> List[str]:
        word += self.eow
        n = len(word)
        best_score = [float('-inf')] * (n + 1)
        best_seg = [None] * (n + 1)
        best_score[0] = 0

        for end in range(1, n + 1):
            for start in range(max(0, end - self.max_subword_length), end):
                sub = word[start:end]
                if sub in self.vocab:
                    score = best_score[start] + self.vocab[sub]
                    if score > best_score[end]:
                        best_score[end] = score
                        best_seg[end] = (start, sub)

        segments = []
        i = n
        while i > 0 and best_seg[i]:
            start, sub = best_seg[i]
            segments.append(sub)
            i = start

        return segments[::-1] if segments else ["<UNK>"]

    def train(self, corpus: List[str], vocab_size: int = 500, num_iterations: int = 5):
        logger.info("🔧 Step 1: Building initial vocab...")
        self.vocab = self.build_initial_vocab(corpus, vocab_size)

        for iteration in range(num_iterations):
            logger.info(f"🔁 EM Iteration {iteration + 1}")
            expected_counts = defaultdict(float)

            for line in corpus:
                for word in line.strip().split():
                    word += self.eow
                    segmentations = self._segmentations(word)
                    if not segmentations:
                        continue

                    Z = sum(
                        math.exp(sum(self.vocab.get(seg, -100) for seg in path))
                        for path in segmentations
                    )

                    for path in segmentations:
                        prob = math.exp(sum(self.vocab.get(seg, -100) for seg in path)) / Z
                        for seg in path:
                            expected_counts[seg] += prob

            total_count = sum(expected_counts.values())
            self.vocab = {
                token: math.log(count / total_count)
                for token, count in expected_counts.items()
                if count > 0
            }

        sorted_vocab = sorted(self.vocab.items(), key=lambda x: -x[1])
        self.token_to_id = {token: i for i, (token, _) in enumerate(sorted_vocab)}
        self.token_to_id["<UNK>"] = len(self.token_to_id)
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        self.trained = True
        logger.info("✅ UML training completed.")

    def encode(self, text: str) -> List[str]:
        if not self.trained:
            raise ValueError("Tokenizer not trained yet.")
        tokens = []
        for word in text.strip().split():
            segmented = self._best_segmentation(word)
            self.token_freqs.update(segmented)
            tokens.extend(segmented)
        return tokens

    def decode(self, token_or_ids: List[Union[int, str]]) -> str:
        if not token_or_ids:
            return ""

        has_int = any(isinstance(t, int) for t in token_or_ids)
        has_str = any(isinstance(t, str) for t in token_or_ids)

        if has_int and has_str:
            logger.warning("Mixed token types detected. Interpreting ints as token IDs and strs as raw tokens.")

        tokens = []
        for t in token_or_ids:
            if isinstance(t, int):
                token = self.id_to_token.get(t, "<UNK>")
            elif isinstance(t, str):
                token = t
            else:
                logger.warning(f"Ignoring unknown token type: {type(t)}")
                continue

            if token in ["<PAD>", "<UNK>", "<START>", "<END>"] or not token:
                continue

            tokens.append(token)

        words = []
        current_word = ""
        for token in tokens:
            if token.endswith(self.eow):
                current_word += token.replace(self.eow, "")
                words.append(current_word)
                current_word = ""
            else:
                current_word += token
        if current_word:
            words.append(current_word)

        return " ".join(words).strip()

    def encode_ids(self, text: str) -> List[int]:
        tokens = self.encode(text)
        unk_id = self.token_to_id.get("<UNK>", 0)
        return [self.token_to_id.get(token, unk_id) for token in tokens]

    def decode_ids(self, ids: List[int]) -> str:
        return self.decode(ids)

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.id_to_token.get(i, "<UNK>") for i in ids]

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id.get(t, self.token_to_id.get("<UNK>", 0)) for t in tokens]

    def save(self, path: str):
        data = {
            "vocab": self.vocab,
            "token_to_id": self.token_to_id
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab = data.get("vocab", {})
        self.token_to_id = data.get("token_to_id", {})
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        self.trained = bool(self.vocab)

        if "<UNK>" not in self.token_to_id:
            unk_id = len(self.token_to_id)
            self.token_to_id["<UNK>"] = unk_id
            self.id_to_token[unk_id] = "<UNK>"

    def save_pickle(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                "vocab": self.vocab,
                "token_to_id": self.token_to_id,
                "id_to_token": self.id_to_token,
                "trained": self.trained
            }, f)
        logger.info(f"Tokenizer saved to {path} (pickle format)")

    def load_pickle(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.vocab = data.get("vocab", {})
        self.token_to_id = data.get("token_to_id", {})
        self.id_to_token = data.get("id_to_token", {})
        self.trained = data.get("trained", False)

        if "<UNK>" not in self.token_to_id:
            unk_id = len(self.token_to_id)
            self.token_to_id["<UNK>"] = unk_id
            self.id_to_token[unk_id] = "<UNK>"

        logger.info(f"Tokenizer loaded from {path} (pickle format)")


# Demo if run directly
if __name__ == "__main__":
    tokenizer = UnigramTokenizer()
    corpus = ["machine learning is powerful", "learning algorithms are amazing"]
    tokenizer.train(corpus)

    tokenizer.save_pickle("uml_tokenizer.pkl")

    # Load from pickle and test
    new_tokenizer = UnigramTokenizer()
    new_tokenizer.load_pickle("uml_tokenizer.pkl")

    test_text = "learning is fun"
    tokens = new_tokenizer.encode(test_text)
    ids = new_tokenizer.encode_ids(test_text)
    decoded = new_tokenizer.decode_ids(ids)

    logger.info(f"[Pickle] Encoded Tokens: {tokens}")
    logger.info(f"[Pickle] Encoded IDs: {ids}")
    logger.info(f"[Pickle] Decoded: {decoded}")
