# Text Summarizer (Extractive + Abstractive)

A Python Streamlit app for summarizing text documents using both **extractive** (TF-IDF) and **abstractive** (DistilBART) techniques. Works with **Wikipedia articles** or **any uploaded text file**. Includes chunking for long texts and optional ROUGE evaluation.

---

## Features

- Summarizes **Wikipedia articles** by topic.  
- Summarizes **uploaded text files** in `.txt` format.  
- Generates **extractive summary** using TF-IDF scoring.  
- Generates **abstractive summary** using **DistilBART model**.  
- Handles **long text** by splitting into chunks.  
- Computes **ROUGE scores** using extractive summary as a pseudo-reference.  
- Allows **download** of abstractive summaries.  

---

## Model

This app uses the [DistilBART CNN model](https://huggingface.co/sshleifer/distilbart-cnn-12-6) from Hugging Face for **abstractive summarization**.

- **Model:** `sshleifer/distilbart-cnn-12-6`  
- **Usage:** Local deployment via `transformers` pipeline.  
- **Note:** You need to download the model once using Hugging Face Hub:

```python
from huggingface_hub import snapshot_download
snapshot_download("sshleifer/distilbart-cnn-12-6", local_dir="local_distilbart_model")
