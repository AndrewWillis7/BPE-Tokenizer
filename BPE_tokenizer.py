
# pip install tqdm datasets hf_xet

import json
import re
import collections
import unicodedata
from tqdm import tqdm
from typing import List, Tuple, Dict

SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>", "<mask>"]

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def get_vocab_from_corpus(corpus: List[str]) -> Dict[Tuple[str,...], int]:
    vocab = collections.Counter()
    for line in corpus:
        line = normalize_text(line)
        words = line.split(" ")
        for w in words:
            if not w:
                continue
            token = tuple(list(w) + ["</w>"])
            vocab[token] += 1
    return vocab

def get_pair_frequencies(vocab: Dict[Tuple[str,...], int]) -> Dict[Tuple[str,str], int]:
    pairs = collections.Counter()
    for token, freq in vocab.items():
        for i in range(len(token)-1):
            pairs[(token[i], token[i+1])] += freq
    return pairs

def merge_pair(vocab, pair):
    a, b = pair
    new_vocab = {}
    for token, freq in vocab.items():
        new_token = []
        i = 0
        while i < len(token):
            if i < len(token)-1 and token[i] == a and token[i+1] == b:
                new_token.append(a + b)
                i += 2
            else:
                new_token.append(token[i])
                i += 1
        # Add merged token to the new vocab dict
        new_vocab[tuple(new_token)] = freq
    return new_vocab

def train_bpe(corpus: List[str], num_merges: int = 10000):
    vocab = get_vocab_from_corpus(corpus)
    merges = []

    print(f"Starting BPE training with {num_merges} merges...")
    pbar = tqdm(range(num_merges), desc="Training BPE", ncols=100)
    
    for i in pbar:
        pairs = get_pair_frequencies(vocab)
        if not pairs:
            break
        most_common, freq = pairs.most_common(1)[0]
        merges.append(most_common)
        vocab = merge_pair(vocab, most_common)

        # Optional live info
        if i % 500 == 0:
            pbar.set_postfix_str(f"Top freq={freq:,}, vocab={len(vocab):,}")

    pbar.close()
    print(f"Finished {len(merges)} merges")

    # Build token list
    token_set = set()
    for token in vocab:
        for sym in token:
            token_set.add(sym)
    token_list = SPECIAL_TOKENS + sorted(token_set - set(SPECIAL_TOKENS))
    token2id = {tok: i for i, tok in enumerate(token_list)}
    return merges, token2id

def save_tokenizer(merges: List[Tuple[str, str]], token2id: Dict[str, int], path_prefix: str):
    """Saves merges.txt and vocab.json"""
    with open(f"{path_prefix}_merges.txt", "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a} {b}\n")
    with open(f"{path_prefix}_vocab.json", "w", encoding="utf-8") as f:
        json.dump(token2id, f, ensure_ascii=False, indent=2)

def load_tokenizer(path_prefix: str):
    """Loads merges.txt and vocab.json"""
    with open(f"{path_prefix}_merges.txt", "r", encoding="utf-8") as f:
        merges = [tuple(line.strip().split()) for line in f if line.strip()]
    with open(f"{path_prefix}_vocab.json", "r", encoding="utf-8") as f:
        token2id = json.load(f)
    return merges, token2id

# Class Setup
class BPETokenizer:
    def __init__(self, merges: List[Tuple[str, str]], token2id: Dict[str, int]):
        self.merges = merges
        self.token2id = token2id
        self.id2token = {i: t for t, i in token2id.items()}
        self.unk = "<unk>"
        self.pad = "<pad>"
        self.eos = "</s>"
        self.bos = "<s>"
        self.mask = "<mask>"

        # map from pair -> merged token
        self.pair2merged = {(a, b): a + b for a, b in merges}

    def _encode_word(self, word: str):
        symbols = list(word) + ["</w>"]
        while True:
            pairs = [(symbols[i], symbols[i+1]) for i in range(len(symbols)-1)]
            merge_candidates = [p for p in pairs if p in self.pair2merged]
            if not merge_candidates:
                break
            # find earliest merge in merges list
            first = min(merge_candidates, key=lambda p: self.merges.index(p))
            merged = self.pair2merged[first]
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols)-1 and (symbols[i], symbols[i+1]) == first:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        # map to ids
        unk_id = self.token2id.get(self.unk)
        return [self.token2id.get(s, unk_id) for s in symbols if s != "</w>"]

    def encode(self, text: str, add_special=True):
        text = normalize_text(text)
        ids = []
        if add_special:
            ids.append(self.token2id.get(self.bos))
        for word in text.split(" "):
            ids.extend(self._encode_word(word))
        if add_special:
            ids.append(self.token2id.get(self.eos))
        return ids

    def decode(self, ids):
        tokens = [self.id2token.get(i, self.unk) for i in ids]
        text = "".join(tokens).replace("</w>", " ")
        return text.strip()
