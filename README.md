# Engagement Analyzer: LLM-Driven Discourse Extraction & Evaluation

> **Note:** If you are looking for a quick overview or want to try the interactive web application, please visit the [Engagement Analyzer Demo Repository](link-to-your-demo-repo-here).

## 📖 Overview
The **Engagement Analyzer** is a computational linguistics research pipeline designed to extract, classify, and evaluate complex rhetorical and discourse features (specifically "Engagement" labels) from unstructured text. 

Developed as part of undergraduate research at Waseda University, this project tackles fundamental challenges in Natural Language Understanding (NLU), including severe data sparsity, semantic text understanding, and strict token boundary detection. The repository houses the core evaluation harnesses, data processing pipelines, and experimental frameworks used to benchmark frontier Large Language Models (LLMs) against traditional discriminative baselines.

## 🎯 Research Objectives
* **Bridging the Lexical Gap:** Moving beyond simple keyword matching to perform deep semantic extraction of rhetorical strategies used by authors to engage readers.
* **Compound AI Evaluation:** Building robust evaluation metrics to assess not just classification accuracy, but the exact token-level boundary extraction of unstructured text snippets.
* **Data Sparsity Mitigation:** Designing synthetic data generation and augmentation pipelines utilizing advanced prompting techniques to train highly accurate sequence taggers despite limited annotated datasets.

## 🧠 Methodology & Architecture

This repository explores two primary architectural approaches to text extraction and classification:

### 1. Discriminative Baselines (Sequence Tagging)
We implemented and fine-tuned transformer-based discriminative models (primarily **RoBERTa**) to establish a rigorous baseline for token classification. This involves:
* High-quality data curation and strict annotation guidelines.
* Optimizing model architectures for strict memory limits and edge-like deployment constraints (CPU-only inference).

### 2. Generative Agentic Workflows (LLMs)
To overcome the limitations of traditional sequence tagging, we integrated frontier generative models (including **Llama 3**). The LLM pipeline focuses on:
* **Few-Shot Chain-of-Thought (CoT) Prompting:** Forcing the model into multi-step reasoning to accurately identify implicit rhetorical features.
* **Structured Generation:** Utilizing strict prompt engineering and parsing mechanisms to ensure the LLM outputs reliable, deterministic JSON structures for downstream evaluation.
* **Evaluation Harnesses:** A custom testing suite to measure LLM grounding, hallucination rates, and token extraction precision against the RoBERTa baselines.

## 🗂️ Repository Structure

* `data_processing/`: Scripts for cleaning, chunking, and formatting unstructured discourse data for both discriminative training and LLM ingestion.
* `models_baseline/`: Training, fine-tuning, and inference scripts for RoBERTa and other baseline sequence taggers.
* `llm_harness/`: The core evaluation framework for generative models, including prompt templates, API integration logic, and output parsers.
* `evaluation/`: Custom metrics for assessing token boundary accuracy, semantic matching, and execution traces.
* `experiments/`: Ablation studies, prompt vulnerability tests, and parallel ranking experiments.

## 🔬 Tech Stack
* **Machine Learning & NLP:** PyTorch, Hugging Face Transformers, RoBERTa, spaCy
* **LLM Integration:** Llama 3, Prompt Engineering (Few-Shot, CoT), Structured Outputs
* **Data & Tooling:** Python, pandas, NumPy, Git LFS (for large weight management)

## 🤝 Context
This research bridges theoretical linguistics with practical, production-ready AI. The frameworks built here demonstrate how to construct eval-driven, compound AI systems that prioritize both output accuracy and system reliability in complex text-processing environments.
