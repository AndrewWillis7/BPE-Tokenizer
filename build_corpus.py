

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
        print(f"⬇️ Loading {name}...")
        try:
            ds = load_dataset(name)
        except Exception as e:
            print(f"❌ Failed to load {name}: {e}")
            continue

        for split in ds.keys():
            data = ds[split]

            # DailyDialog
            if name == "daily_dialog" and "dialog" in data.features:
                for conv in data["dialog"]:
                    for utter in conv:
                        if isinstance(utter, str):
                            corpus.append(normalize_text(utter))

            # PersonaChat
            elif name == "persona_chat":
                for ex in data:
                    for utt in ex.get("utterances", []):
                        for candidate in utt.get("candidates", []):
                            corpus.append(normalize_text(candidate))
                    for h in ex.get("history", []):
                        corpus.append(normalize_text(h))

            # Blended Skill Talk
            elif name == "blended_skill_talk":
                for ex in data:
                    for field in ["previous_utterance", "free_messages", "guided_messages"]:
                        val = ex.get(field)
                        if isinstance(val, str):
                            corpus.append(normalize_text(val))
                        elif isinstance(val, list):
                            corpus.extend([normalize_text(v) for v in val if isinstance(v, str)])

            # Generic fallback
            else:
                text_keys = [k for k in data.features if "text" in k or "utter" in k or "dialog" in k]
                for k in text_keys:
                    for v in data[k]:
                        if isinstance(v, str):
                            corpus.append(normalize_text(v))

        print(f"✅ Loaded {len(corpus):,} total lines so far from {name}")

    print(f"\n🎯 Final corpus size: {len(corpus):,} lines")
    return corpus
