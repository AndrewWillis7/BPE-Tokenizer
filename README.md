# 🧠 Conversational Corpus Builder + BPE Tokenizer

A lightweight Python project for building a **large conversational text corpus** from Hugging Face datasets and preparing it for **Byte-Pair Encoding (BPE) tokenization**.

This project combines multiple open-source dialogue datasets (like *DailyDialog* and *PersonaChat*) into one normalized `.txt` file suitable for tokenizer training, fine-tuning, or LLM data preprocessing.

---

## 🚀 Features

- 🔹 Combine multiple conversational datasets into one unified corpus  
- 🔹 Automatically normalizes and cleans text  
- 🔹 Supports any Hugging Face dataset (auto-detects text fields)  
- 🔹 Saves the final corpus as a plain text file (`corpus.txt`)  
- 🔹 Easily extendable for new datasets  
- 🔹 Ready for use with custom BPE tokenizers or Hugging Face’s `tokenizers` library  

---
