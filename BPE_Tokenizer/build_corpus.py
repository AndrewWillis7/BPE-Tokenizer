import datasets
from datasets import load_dataset
from BPE_tokenizer import normalize_text

def build_conversational_corpus(dataset_names):
    """
    Build a combined conversational corpus from one or more Hugging Face datasets.

    Args:
        dataset_names (list[str]): List of dataset names to include, e.g. ["daily_dialog", "persona_chat"]

    Returns:
        list[str]: Flattened list of normalized text utterances.
    """
    corpus = []

    for name in dataset_names:
        print(f"Loading {name}...")
        try:
            ds = load_dataset(name)
        except Exception as e:
            print(f"Failed to load {name}: {e}")
            continue

        for split in ds.keys():
            data = ds[split]
            # Generic fallback
            text_keys = [k for k in data.features if "text" in k or "utterances" in k or "dialogue" in k or "prompt" in k]
            for k in text_keys:
                for v in data[k]:
                    if isinstance(v, str):
                        corpus.append(normalize_text(v))

        print(f"Loaded {len(corpus):,} total lines so far from {name}")

        # Save to file
    with open("corpus.txt", "w", encoding="utf-8") as f:
        for line in corpus:
            f.write(line + "\n")

    print(f"\nFinal corpus size: {len(corpus):,} lines")
    return corpus
