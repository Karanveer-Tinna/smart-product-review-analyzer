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

## Data Preprocessing

The raw Amazon review dataset is first cleaned and prepared to create a balanced and manageable dataset for model training. Reviews are filtered based on length, retaining only reviews containing **50–500 characters** to remove extremely short or excessively long inputs. The dataset is then **stratified by star rating** to maintain a balanced representation of different rating classes.

A fixed random seed is used to ensure reproducibility. The resulting dataset contains up to **20,000 training samples, 2,000 validation samples, and 2,000 test samples**. Each split is processed independently before being saved to disk in Hugging Face Dataset format.

### Preprocessing Pipeline

```text
Amazon Reviews Dataset
          ↓
Review Length Filtering
      (50–500 chars)
          ↓
Stratified Sampling
      (by star rating)
          ↓
Train / Validation / Test
          ↓
Prepared Review Dataset
```

## Data Augmentation & Label Generation

Since the original dataset does not provide reference summaries or explicit bias labels, additional supervision is generated automatically using pretrained NLP models and rule-based heuristics.

### Summary Generation

**FLAN-T5-base** is used to generate a concise, factual summary for each review. Reviews are converted into prompts instructing the model to focus on the primary praise or complaint while avoiding direct repetition. The generated summaries serve as reference targets for fine-tuning TinyLlama.

### Bias Label Generation

Potential review bias is identified using a combination of **BERT-based sentiment analysis and rule-based linguistic features**.

The `nlptown/bert-base-multilingual-uncased-sentiment` model predicts the sentiment rating of each review. This prediction is compared with the review's original star rating. A large disagreement between the two ratings contributes to the bias score.

Additional heuristic signals include:

* Strong positive or negative expressions
* Toxic or highly emotional language
* Strong rejection phrases such as *"do not buy"* or *"waste of money"*
* Contradictory positive and negative language
* Mismatch between extreme language and the assigned star rating
* Issue-related words combined with strong rejection statements

The individual signals are combined into a **bias score**. Reviews with a score of **2.0 or higher are assigned a bias label of `1`**, while the remaining reviews receive a label of `0`.

```text
                 Product Review
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
     FLAN-T5-base             BERT Sentiment
          │                         │
          ▼                         ▼
      Summary              Predicted Rating
                                    │
                                    ▼
                         Compare with Star Rating
                                    │
                                    ▼
                         Linguistic Heuristics
                                    │
                                    ▼
                              Bias Score
                                    │
                              ┌─────┴─────┐
                              ▼           ▼
                         Score ≥ 2    Score < 2
                              │           │
                              ▼           ▼
                         Bias = 1     Bias = 0
```

The final augmented dataset therefore contains the original review and star rating along with a generated **summary** and **bias label**, providing the supervised training targets required for the TinyLlama fine-tuning stage.

## Fine-Tuning Methodology

The project fine-tunes **TinyLlama-1.1B-Chat-v1.0** using **Supervised Fine-Tuning (SFT)** to adapt the pretrained language model to product-review summarization and bias detection. The augmented dataset contains reviews paired with generated summaries and bias labels, allowing the model to learn both tasks within a unified instruction-following framework.

The training process uses **Hugging Face Transformers** and PyTorch. To make fine-tuning feasible on limited GPU hardware, memory-efficient techniques such as **bfloat16 precision and quantization** are incorporated.

### Training Configuration

| Parameter          | Configuration                            |
| ------------------ | ---------------------------------------- |
| Base Model         | TinyLlama-1.1B-Chat-v1.0                 |
| Fine-Tuning Method | Supervised Fine-Tuning (SFT)             |
| Framework          | PyTorch + Hugging Face Transformers      |
| Training Data      | Augmented Amazon Reviews                 |
| Tasks              | Summarization + Bias Detection           |
| Precision          | bfloat16                                 |
| Optimization       | Quantization / Memory-efficient Training |

## Training Pipeline

The training pipeline converts the augmented review dataset into instruction-style examples before fine-tuning TinyLlama.

```text
                 Augmented Dataset
                        │
                        ▼
              Review + Summary +
                 Bias Label
                        │
                        ▼
             Instruction Formatting
                        │
                        ▼
                 Tokenization
                        │
                        ▼
              TinyLlama-1.1B-Chat
                        │
                        ▼
            Supervised Fine-Tuning
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
   Review Summarization          Bias Detection
          │                           │
          └─────────────┬─────────────┘
                        ▼
                Fine-Tuned Model
                        │
                        ▼
              Quantized Inference
                        │
                        ▼
                Evaluation
```

### Training Stages

1. **Dataset Preparation:** The augmented reviews, generated summaries, and bias labels are loaded and formatted into instruction-response examples.
2. **Tokenization:** Review prompts and expected responses are tokenized using the TinyLlama tokenizer with appropriate truncation and padding.
3. **Supervised Fine-Tuning:** TinyLlama is trained on the prepared examples to learn the relationship between product reviews and their corresponding summaries and bias classifications.
4. **Memory Optimization:** bfloat16 precision and quantization techniques reduce GPU memory consumption and enable training on a limited GPU.
5. **Model Validation:** The validation split is used to monitor model performance during training and identify potential overfitting.
6. **Evaluation:** The resulting model is evaluated on the held-out test set using ROUGE and BLEU for summarization and Precision, Recall, and F1-score for bias detection.


## Evaluation Results — Summarization

The summarization performance of the **fine-tuned TinyLlama model** was evaluated against the **base TinyLlama model** using ROUGE and BLEU scores on the test set. The results demonstrate a substantial improvement in the quality and relevance of generated product-review summaries after fine-tuning.

### ROUGE Results

| Model                    |    ROUGE-1 |    ROUGE-2 |    ROUGE-L | ROUGE-Lsum |
| ------------------------ | ---------: | ---------: | ---------: | ---------: |
| Base TinyLlama           |     0.1904 |     0.0645 |     0.1553 |     0.1554 |
| **Fine-Tuned TinyLlama** | **0.5092** | **0.4448** | **0.5039** | **0.5012** |

The fine-tuned model significantly outperforms the baseline across all ROUGE metrics. In particular, **ROUGE-2 increased from 0.0645 to 0.4448**, indicating a substantial improvement in matching meaningful two-word sequences from the reference summaries.

### BLEU Results

| Model                    |       BLEU |     1-gram |     2-gram |     3-gram |     4-gram |
| ------------------------ | ---------: | ---------: | ---------: | ---------: | ---------: |
| Base TinyLlama           |     0.0514 |     0.1867 |     0.0580 |     0.0316 |     0.0204 |
| **Fine-Tuned TinyLlama** | **0.5492** | **0.5905** | **0.5444** | **0.5344** | **0.5295** |

The BLEU score increased from **0.0514 to 0.5492**, showing that fine-tuning substantially improved the model's ability to generate summaries that align with the reference summaries.

### Overall Improvement

| Metric     | Baseline | Fine-Tuned | Improvement |
| ---------- | -------: | ---------: | ----------: |
| ROUGE-1    |   0.1904 | **0.5092** |    +167.44% |
| ROUGE-2    |   0.0645 | **0.4448** |    +589.61% |
| ROUGE-L    |   0.1553 | **0.5039** |    +224.40% |
| ROUGE-Lsum |   0.1554 | **0.5012** |    +222.52% |
| BLEU       |   0.0514 | **0.5492** |    +968.48% |

Overall, the evaluation indicates that **domain-specific supervised fine-tuning substantially improves TinyLlama's product-review summarization capability** compared with the base model. The fine-tuned model produces outputs that have considerably greater lexical and n-gram overlap with the generated reference summaries.
