import spacy
from spacy.tokens import DocBin
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot import plt
import numpy as np
import os

def load_sentences_by_category(file_path, target_category, spans_key="sc"):
    nlp = spacy.blank("en")
    doc_bin = DocBin().from_disk(file_path)
    docs = list(doc_bin.get_docs(nlp.vocab))
    
    filtered_texts = []
    for doc in docs:
        # Check if the target category exists in the document's spans
        if any(span.label_ == target_category for span in doc.spans.get(spans_key, [])):
            filtered_texts.append(doc.text)
    return filtered_texts

def main():
    # Change this to whatever category you want to analyze (e.g., "SOURCES", "JUSTIFYING")
    TARGET_CATEGORY = "SOURCES"
    
    print(f"Loading datasets and filtering for {TARGET_CATEGORY}...")
    gold_file = "data/train.spacy"
    silver_file = "data/pseudo_labeled_training_corpus.spacy"
    llm_file = "data/synthetic_pseudo_labeled_few_shot_v3.spacy"

    gold_texts = load_sentences_by_category(gold_file, TARGET_CATEGORY)
    silver_texts = load_sentences_by_category(silver_file, TARGET_CATEGORY)
    llm_texts = load_sentences_by_category(llm_file, TARGET_CATEGORY)
    
    # Subsample to balance the graph visually
    SAMPLE_SIZE = min(len(gold_texts), len(silver_texts), len(llm_texts), 1000) 
    
    print(f"Sampling {SAMPLE_SIZE} sentences from each dataset for {TARGET_CATEGORY}...")
    gold_texts = np.random.choice(gold_texts, SAMPLE_SIZE, replace=False)
    silver_texts = np.random.choice(silver_texts, SAMPLE_SIZE, replace=False)
    llm_texts = np.random.choice(llm_texts, SAMPLE_SIZE, replace=False)

    print("Generating MiniLM embeddings...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    all_texts = list(gold_texts) + list(silver_texts) + list(llm_texts)
    embeddings = embedder.encode(all_texts, show_progress_bar=True)

    print("Fitting t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    embeddings_2d = tsne.fit_transform(embeddings)

    gold_2d = embeddings_2d[:SAMPLE_SIZE]
    silver_2d = embeddings_2d[SAMPLE_SIZE:SAMPLE_SIZE*2]
    llm_2d = embeddings_2d[SAMPLE_SIZE*2:]

    os.makedirs("visualizations", exist_ok=True)
    alpha_val = 0.6
    dot_size = 15

    # --- PLOT COMBINED ---
    plt.figure(figsize=(10, 8))
    plt.scatter(gold_2d[:, 0], gold_2d[:, 1], c='blue', label='Gold (Human)', alpha=alpha_val, s=dot_size)
    plt.scatter(silver_2d[:, 0], silver_2d[:, 1], c='green', label='Silver (Real Corpus)', alpha=alpha_val, s=dot_size)
    plt.scatter(llm_2d[:, 0], llm_2d[:, 1], c='red', label='LLM (Synthetic)', alpha=alpha_val, s=dot_size)
    plt.title(f't-SNE Overlay: Semantic Distribution of {TARGET_CATEGORY}')
    plt.legend()
    plt.savefig(f'visualizations/tsne_combined_{TARGET_CATEGORY}.png', dpi=300)
    plt.close()
    
    print(f"[SUCCESS] Saved combined t-SNE graph for {TARGET_CATEGORY}.")

if __name__ == "__main__":
    main()
