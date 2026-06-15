# Smart Product Review Analyzer

Fine-tuned TinyLlama model that summarizes Amazon product reviews and detects review bias. Uses supervised fine-tuning, data augmentation, ROUGE/BLEU evaluation, and an intuitive interface for review analysis.


## Features

- **Abstractive Review Summarization** – Generates concise, human-readable summaries from lengthy and noisy product reviews.
- **Review Bias Detection** – Identifies potentially biased, misleading, or emotionally exaggerated reviews.
- **Fine-Tuned Large Language Model** – Adapts TinyLlama-1.1B-Chat to the product review domain using Supervised Fine-Tuning (SFT).
- **Automated Data Augmentation Pipeline** – Creates ground-truth summaries and bias labels using FLAN-T5 and BERT-based models.
- **Quantitative Performance Evaluation** – Measures improvements using ROUGE, BLEU, Precision, Recall, and F1-score metrics.
- **Efficient Consumer-Grade Deployment** – Utilizes quantization techniques for training and inference on limited hardware resources.
- **E-Commerce Intelligence Tool** – Helps consumers and analysts quickly extract insights from large volumes of product reviews.


## Tech Stack

### Programming Language
- Python

### Deep Learning Frameworks
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets

### Machine Learning Models
- TinyLlama-1.1B-Chat-v1.0 (Fine-Tuned LLM)
- google/flan-t5-base (Summary Generation)
- nlptown/bert-base-multilingual-uncased-sentiment (Bias Label Generation)

### Data Processing & Analysis
- Pandas
- NumPy
- Regular Expressions (re)

### Optimization & Fine-Tuning
- BitsAndBytes
- bfloat16 Precision
- Supervised Fine-Tuning (SFT)

### Development Environment
- Jupyter Notebook
- Kaggle Notebooks
- NVIDIA Tesla T4 GPU
- CUDA

### Evaluation Metrics
- ROUGE-1
- ROUGE-2
- ROUGE-L
- BLEU Score
- Precision
- Recall
- F1 Score