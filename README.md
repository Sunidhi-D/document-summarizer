# document-summarizer
A Python Streamlit app for summarizing text documents using both extractive (TF-IDF) and abstractive (DistilBART) techniques. Works with Wikipedia articles or any uploaded text file. Includes chunking for long texts and optional ROUGE evaluation.

# Text Summarizer (Extractive + Abstractive)

A Python Streamlit app for summarizing text documents using both **extractive** (TF-IDF) and **abstractive** (DistilBART) techniques. Works with **Wikipedia articles** or **any uploaded text file**. Includes chunking for long texts and optional ROUGE evaluation.

---

## Features

- Summarizes **Wikipedia articles** by topic.  
- Summarizes **uploaded text files** in `.txt` format.  
- Generates **extractive summary** using TF-IDF scoring.  
- Generates **abstractive summary** using **local DistilBART model**.  
- Handles **long text** by splitting into chunks.  
- Computes **ROUGE scores** using extractive summary as a pseudo-reference.  
- Allows **download** of abstractive summaries.  

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/text-summarizer.git
cd text-summarizer
