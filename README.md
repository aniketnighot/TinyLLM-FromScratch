# TinyLLM-FromScratch 🚀
*A character-level Transformer Decoder LLM built entirely from scratch using PyTorch.*

---

## 📚 Project Overview

**TinyLLM-FromScratch** is a minimal yet powerful **Transformer Decoder model** that generates text **character-by-character** — without using any pre-built architectures.  
This project demonstrates a complete understanding of **LLMs**, **Self-Attention**, and **Transformer architecture**, all implemented **from first principles**.

Trained on song lyrics, the model learns to predict the next character based on previous characters, showcasing the power of attention mechanisms even at a small scale.

---

## 🏆 Highlights
- ✍️ Full **Transformer Decoder** implemented manually in PyTorch
- 🧠 Custom-built **Self-Attention**, **Multi-Head Attention**, **Feed Forward Networks**
- 🔥 **Character-level** language modeling without any external tokenizers
- 🛠️ **Training loop**, **batching**, and **generation** fully coded from scratch
- 📖 Trained on real Hindi lyrics dataset for creative text generation
- 📦 Fully modular and easy-to-extend project structure

---

## 🔥 Tech Stack
- Python 3.10+
- PyTorch
- Numpy

---

## 🛠️ How It Works
1. Raw text data is read and unique characters are tokenized.
2. Inputs are fed into an **embedding layer** + **positional encoding**.
3. Data passes through multiple layers of:
   - **Masked Self-Attention**
   - **Multi-Head Attention**
   - **Feed Forward Networks**
4. Final outputs predict the **next character** probabilities.
5. Sampling from output allows **new text generation**.

---

## 🗂️ Folder Structure
```
TinyLLM-FromScratch/
├── biogram_model/
│   ├── model.py        # Transformer Model
│   ├── utils.py        # Tokenizer and helper functions
├── hindi_song_lyrics.txt  # Training dataset
├── train.py            # Model training script
├── notebook/
│   └── GPT_From_Scratch.ipynb  # Full Colab Notebook
├── README.md
├── requirements.txt
├── LICENSE
```

 
---

## Quickstart
1. **Clone this repository**
   
   ```bash
   git clone https://github.com/aniketnighot/TinyLLM-FromScratch.git
   cd TinyLLM-FromScratch

3. **Install Dependencies**
   
   ```bash
   pip install -r requirements.txt

4. **Train the Model**
   
   ```bash
   python train.py

5. **Generate New Text After training, the model will generate ~500 characters of new Hindi lyrics based on learned patterns.**

## 📈 Sample Output

tera naam lene ki aadat ban gayi hai,
meri raaton ki chahat ban gayi hai,
tera intezaar rehne laga hai har pal,
meri khamoshi ki baat ban gayi hai...

## 🧠 Future Improvements


1. Train on larger multilingual datasets
2. Add support for word-level tokenization
3. Scale model size (more heads, more layers)




## Acknowledgments
Inspired by Andrej Karpathy's teachings and philosophy.

Self-implemented for deep understanding of LLMs and Transformer models.


