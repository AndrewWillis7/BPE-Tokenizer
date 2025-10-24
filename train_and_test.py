# train_and_test.py
from BPE_Tokenizer.BPE_tokenizer import train_bpe, save_tokenizer, load_tokenizer, BPETokenizer

# Train
corpus = [
    "Hello world!",
    "Hello there world!",
    "This is a tiny BPE tokenizer test."
]

merges, token2id = train_bpe(corpus, num_merges=100)
save_tokenizer(merges, token2id, "test_bpe")

# Reload
merges, token2id = load_tokenizer("test_bpe")
tokenizer = BPETokenizer(merges, token2id)

# Encode/decode test
text = "Hello world!"
ids = tokenizer.encode(text)
print("Encoded:", ids)
print("Decoded:", tokenizer.decode(ids))
