# train_and_test.py
from BPE_tokenizer import train_bpe, save_tokenizer, load_tokenizer, BPETokenizer
from build_corpus import build_conversational_corpus

# Train
corpus = build_conversational_corpus(["scryptiam/anime-waifu-personality-chat",
                                      "fka/awesome-chatgpt-prompts",
                                      "humarin/chatgpt-paraphrases"])

merges, token2id = train_bpe(corpus, num_merges=100)
save_tokenizer(merges, token2id, "test_bpe")

# Reload
merges, token2id = load_tokenizer("test_bpe")
tokenizer = BPETokenizer(merges, token2id)

# Encode/decode test
text = "I-I love compsci@ ugh!"
ids = tokenizer.encode(text)
print("Encoded:", ids)
print("Decoded:", tokenizer.decode(ids))
