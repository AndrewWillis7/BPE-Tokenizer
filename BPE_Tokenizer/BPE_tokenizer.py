
# pip install tqdm datasets hf_xet setuptools

import json
import re
import collections
import unicodedata
from tqdm import tqdm
from typing import List, Tuple, Dict

SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>", "<mask>"]

def normalize_text(s: str) -> str:
    """
    Normalize input text by applying Unicode normalization and whitespace cleanup.

    Args:
        s (str): Input string to normalize.

    Returns:
        str: Normalized string.
    """
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def get_vocab_from_corpus(corpus: List[str]) -> Dict[Tuple[str,...], int]:
    """
    Build a vocabulary from a list of text lines.

    Args:
        corpus (List[str]): List of text lines.

    Returns:
        Dict[Tuple[str,...], int]: Vocabulary with token tuples as keys and frequencies as values.
    """
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
    """
    Calculate frequencies of adjacent symbol pairs in the vocabulary.

    Args:
        vocab (Dict[Tuple[str,...], int]): Vocabulary with token tuples as keys and frequencies as values.

    Returns:
        Dict[Tuple[str,str], int]: Frequencies of adjacent symbol pairs.
    """
    pairs = collections.Counter()
    for token, freq in vocab.items():
        for i in range(len(token)-1):
            pairs[(token[i], token[i+1])] += freq
    return pairs

def merge_pair(vocab, pair):
    """
    Merge a given pair of symbols in the vocabulary.

    Args:
        vocab (Dict[Tuple[str,...], int]): Vocabulary with token tuples as keys and frequencies as values.
        pair (Tuple[str, str]): Pair of symbols to merge.

    Returns:
        Dict[Tuple[str,...], int]: Updated vocabulary with merged pairs.
    """
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
    """
    Trains BPE tokenizer on the given corpus.

    Args:
        corpus (List[str]): List of text lines.
        num_merges (int): Number of merge operations to perform.
    Returns:
        Tuple[List[Tuple[str, str]], Dict[str, int]]: List of merges and token to ID mapping.
    """
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
    """
    Saves merges.txt and vocab.json

    Args:
        merges (List[Tuple[str, str]]): List of merges.
        token2id (Dict[str, int]): Token to ID mapping.
    Returns:
        None
    """
    with open(f"{path_prefix}_merges.txt", "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a} {b}\n")
    with open(f"{path_prefix}_vocab.json", "w", encoding="utf-8") as f:
        json.dump(token2id, f, ensure_ascii=False, indent=2)

def load_tokenizer(path_prefix: str):
    """
    Loads merges.txt and vocab.json

    Args:
        path_prefix (str): Prefix path for the tokenizer files.

    Returns:
        Tuple[List[Tuple[str, str]], Dict[str, int]]: List of merges and token to ID mapping.
    """
    with open(f"{path_prefix}_merges.txt", "r", encoding="utf-8") as f:
        merges = [tuple(line.strip().split()) for line in f if line.strip()]
    with open(f"{path_prefix}_vocab.json", "r", encoding="utf-8") as f:
        token2id = json.load(f)
    return merges, token2id

# Class Setup
class BPETokenizer:
    """
    Byte Pair Encoding (BPE) Tokenizer class.

    Args:
        merges (List[Tuple[str, str]]): List of merges.
        token2id (Dict[str, int]): Token to ID mapping.

    Returns:
        None
    """
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
        """
        Encodes a single word into BPE tokens.

        Args:
            word (str): The word to encode.

        Returns:
            List[int]: List of token IDs.
        """
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
        """
        Encodes a text string into BPE tokens.

        Args:
            text (str): The text to encode.
            add_special (bool): Whether to add special tokens (BOS, EOS).

        Returns:
            List[int]: List of token IDs.
        """
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
        """
        Decodes a list of token IDs back into a string.

        Args:
            ids (List[int]): List of token IDs.

        Returns:
            str: Decoded string.
        """
        tokens = [self.id2token.get(i, self.unk) for i in ids]
        text = "".join(tokens).replace("</w>", " ")
        return text.strip()
