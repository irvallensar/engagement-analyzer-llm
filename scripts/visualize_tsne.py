import spacy
from spacy.tokens import DocBin
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os

def load_sentences_by_category(file_path, target_category, spans_key="sc"):
    nlp = spacy.blank("en")
    doc_bin = DocBin().from_disk(file_path)
    docs = list(doc_bin.get_docs(nlp.vocab))
    
    filtered_texts = []
    for doc in docs:
        if any(span.label_ == target_category for span in doc.spans.get(spans_key, [])):
            filtered_texts.append(doc.text)
    return filtered_texts

def main():
    # Change this to whatever category you want to analyze (e.g., "SOURCES", "JUSTIFYING")
    TARGET_CATEGORY = "SOURCES"
    SAMPLE_MAX = 2000
    
    print(f"Loading datasets and filtering for {TARGET_CATEGORY}...")
    gold_file = "data/train.spacy"
    silver_file = "data/pseudo_labeled_training_corpus.spacy"
    llm_file = "data/synthetic_pseudo_labeled_few_shot_v3.spacy"

    gold_texts = load_sentences_by_category(gold_file, TARGET_CATEGORY)
    silver_texts = load_sentences_by_category(silver_file, TARGET_CATEGORY)
    llm_texts = load_sentences_by_category(llm_file, TARGET_CATEGORY)
    
    print(f"Sampling up to {SAMPLE_MAX} sentences from each dataset for {TARGET_CATEGORY}...")
    # This samples up to 2000, or the max available if a dataset has fewer than 2000
    gold_texts = np.random.choice(gold_texts, min(SAMPLE_MAX, len(gold_texts)), replace=False)
    silver_texts = np.random.choice(silver_texts, min(SAMPLE_MAX, len(silver_texts)), replace=False)
    llm_texts = np.random.choice(llm_texts, min(SAMPLE_MAX, len(llm_texts)), replace=False)

    len_g = len(gold_texts)
    len_s = len(silver_texts)
    len_l = len(llm_texts)

    print(f"Final counts - Gold: {len_g}, Silver: {len_s}, LLM: {len_l}")

    print("Generating MiniLM embeddings...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    all_texts = list(gold_texts) + list(silver_texts) + list(llm_texts)
    embeddings = embedder.encode(all_texts, show_progress_bar=True)

    print("Fitting t-SNE algorithm...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    embeddings_2d = tsne.fit_transform(embeddings)

    gold_2d = embeddings_2d[:len_g]
    silver_2d = embeddings_2d[len_g:len_g + len_s]
    llm_2d = embeddings_2d[len_g + len_s:]

    os.makedirs("visualizations", exist_ok=True)
    alpha_val = 0.6
    dot_size = 15

    # Determine global axis limits so all 4 graphs share the exact same visual frame
    x_min, x_max = embeddings_2d[:, 0].min() - 5, embeddings_2d[:, 0].max() + 5
    y_min, y_max = embeddings_2d[:, 1].min() - 5, embeddings_2d[:, 1].max() + 5

    print("Plotting individual and combined graphs...")

    # --- PLOT 1: COMBINED ---
    plt.figure(figsize=(10, 8))
    plt.scatter(gold_2d[:, 0], gold_2d[:, 1], c='blue', label='Gold (Human)', alpha=alpha_val, s=dot_size)
    plt.scatter(silver_2d[:, 0], silver_2d[:, 1], c='green', label='Silver (Real Corpus)', alpha=alpha_val, s=dot_size)
    plt.scatter(llm_2d[:, 0], llm_2d[:, 1], c='red', label='LLM (Synthetic)', alpha=alpha_val, s=dot_size)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title(f't-SNE Overlay: Semantic Distribution of {TARGET_CATEGORY}')
    plt.legend()
    plt.savefig(f'visualizations/tsne_1_combined_{TARGET_CATEGORY}.png', dpi=300)
    plt.close()

    # --- PLOT 2: GOLD ONLY ---
    plt.figure(figsize=(10, 8))
    plt.scatter(gold_2d[:, 0], gold_2d[:, 1], c='blue', alpha=alpha_val, s=dot_size)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title(f't-SNE: Internal Clusters of Gold Data ({TARGET_CATEGORY})')
    plt.savefig(f'visualizations/tsne_2_gold_{TARGET_CATEGORY}.png', dpi=300)
    plt.close()

    # --- PLOT 3: SILVER ONLY ---
    plt.figure(figsize=(10, 8))
    plt.scatter(silver_2d[:, 0], silver_2d[:, 1], c='green', alpha=alpha_val, s=dot_size)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title(f't-SNE: Internal Clusters of Silver Data ({TARGET_CATEGORY})')
    plt.savefig(f'visualizations/tsne_3_silver_{TARGET_CATEGORY}.png', dpi=300)
    plt.close()

    # --- PLOT 4: LLM ONLY ---
    plt.figure(figsize=(10, 8))
    plt.scatter(llm_2d[:, 0], llm_2d[:, 1], c='red', alpha=alpha_val, s=dot_size)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title(f't-SNE: Internal Clusters of LLM Data ({TARGET_CATEGORY})')
    plt.savefig(f'visualizations/tsne_4_llm_{TARGET_CATEGORY}.png', dpi=300)
    plt.close()
    
    print(f"[SUCCESS] Saved all 4 t-SNE graphs for {TARGET_CATEGORY}.")

if __name__ == "__main__":
    main()
