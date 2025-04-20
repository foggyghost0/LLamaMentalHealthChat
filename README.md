# LLamaMentalHealthChat

This project fine-tunes Meta's Llama-3.1-8B-Instruct model for mental health conversational applications using QLoRA PEFT and hyperparameter optimization (HPO) with Optuna. 

It includes training, evaluation, and metrics tailored for empathetic and safe dialogue generation.

## Features

- **Efficient Fine-Tuning**: Uses QLoRA (4-bit quantization) for memory-efficient training.
- **Hyperparameter Optimization**: Optuna integration for tuning learning rate and batch size.
- **Comprehensive Evaluation**:
  - Standard NLP metrics (ROUGE, BERTScore).
  - Toxicity detection (`detoxify`).
  - Sentiment alignment and lexical diversity.
  - Custom empathy metrics (keyword-based engagement, normalization, and safety checks).
- **WandB Integration**: Logs training and evaluation metrics.
- **Dataset**: Preprocesses the [thu-coai/augesc](https://huggingface.co/datasets/thu-coai/augesc) dataset into Llama-3 chat format.
