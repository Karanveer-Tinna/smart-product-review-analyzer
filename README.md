# Smart Product Review Analyzer

Fine-tuned TinyLlama model that summarizes Amazon product reviews and detects review bias. Uses supervised fine-tuning, data augmentation, ROUGE/BLEU evaluation, and an intuitive interface for review analysis.

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