Medical Device Q&A Fine-Tuning Pipeline
Overview
This project implements a complete end-to-end fine-tuning workflow for the DistilGPT-2 language model in the medical device Q&A domain. It includes:

Domain-specific prompt engineering

LoRA-based parameter-efficient fine-tuning

Automated benchmarking (pre- and post-training) with BERTScore

Custom Gradio interface for interactive Q&A

Prototype hallucination detection

The pipeline uses curated training data (medical_device_data_v2.json) and a benchmarking set (benchmarking_data_100.json) to optimise model accuracy and reliability.

Features
Data Preparation

Loads JSON datasets for training and benchmarking

Formats Q&A pairs using a domain-specific prompt template

Tokenizes data with the Hugging Face tokenizer

Prompt Engineering

Structured prompt enforces:

Evidence-based responses

Manufacturer guideline citations

Avoidance of clinical recommendations

Fine-Tuning

Uses LoRA for efficient adaptation

Optimised training parameters for stability and accuracy

Tracks and saves best model checkpoint

Evaluation

Pre- and post-fine-tuning benchmarks

Quantitative: BERTScore Precision, Recall, F1

Qualitative: side-by-side answer comparisons

Interactive Gradio UI

Input: question, device type, clinical context, and source

Output: fact-based answer + hallucination detection status

Live demo capability for stakeholders

Installation
Run the following in Google Colab (Pro recommended for GPU):

bash
Copy
Edit
pip install transformers[torch] datasets accelerate peft bitsandbytes evaluate torch
pip install scikit-learn matplotlib seaborn bert-score gradio
Usage
1. Fine-Tuning the Model
Place medical_device_data_v2.json and benchmarking_data_100.json in your working directory.

Run all cells in the notebook to:

Pre-train benchmark evaluation

Fine-tune DistilGPT-2 with LoRA

Post-train evaluation and results comparison

2. Launching the Gradio App
The final section of the script builds a Gradio interface:

python
Copy
Edit
demo.launch()
This will open a browser UI where you can test the fine-tuned model interactively.
