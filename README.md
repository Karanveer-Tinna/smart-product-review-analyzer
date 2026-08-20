# Smart Product Review Analyzer

Fine-tuned TinyLlama model that summarizes Amazon product reviews and detects review bias. Uses supervised fine-tuning, data augmentation, ROUGE/BLEU evaluation, and an intuitive interface for review analysis.

## Project Overview

Smart Product Review Analyzer is an AI-powered tool that analyzes Amazon product reviews using a fine-tuned TinyLlama model. It generates concise summaries of lengthy reviews and identifies potentially biased or emotionally exaggerated content, helping consumers quickly understand product feedback and make informed purchasing decisions.

## Problem Statement

Online products often have hundreds or thousands of reviews, making it difficult for consumers to identify useful information efficiently. Reviews may also contain excessive emotional language, misleading claims, or potential bias. This project aims to develop an efficient NLP system that automatically summarizes product reviews and detects potential review bias while remaining lightweight enough for deployment on consumer-grade hardware.


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

## System Architecture

The Smart Product Review Analyzer follows a multi-stage NLP pipeline that processes raw product reviews, generates training data, fine-tunes TinyLlama, and performs review analysis.

```text
                    Amazon Product Reviews
                             │
                             ▼
                  Data Preprocessing
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
           FLAN-T5 Base              BERT Sentiment
        Summary Generation          / Bias Signals
                │                         │
                └────────────┬────────────┘
                             ▼
                    Training Dataset
                             │
                             ▼
                  TinyLlama-1.1B-Chat
                    Supervised Fine-Tuning
                             │
                             ▼
                       Quantization
                             │
                             ▼
                    Fine-Tuned Model
                             │
                             ▼
                  ┌──────────┴──────────┐
                  ▼                     ▼
             Summarization        Bias Detection
                  │                     │
                  └──────────┬──────────┘
                             ▼
                     Analysis Results
```

### Pipeline Components

* **Data Preprocessing:** Cleans and prepares raw product reviews for training and inference.
* **Data Augmentation:** Uses FLAN-T5 to generate reference summaries and a BERT-based sentiment model to assist with bias-label generation.
* **Model Fine-Tuning:** Fine-tunes TinyLlama-1.1B-Chat using Supervised Fine-Tuning (SFT) on the generated training data.
* **Optimization:** Uses bfloat16 precision and quantization techniques to reduce memory requirements and enable efficient execution on consumer-grade hardware.
* **Inference:** The fine-tuned model processes new reviews and produces a concise summary along with a potential bias classification.
* **Evaluation:** Summarization is evaluated using ROUGE and BLEU, while bias detection is evaluated using Precision, Recall, and F1-score.
