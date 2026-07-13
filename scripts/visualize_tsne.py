import spacy
from spacy.tokens import DocBin
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot import plt
import numpy as np
import os

def load_sentences_from_spacy(file_path):
    nlp = spacy.blank("en")
    doc_bin = DocBin().from_disk(file_path)
    docs = list(doc_bin.get_docs(nlp.vocab))
    return [doc.text for doc in docs]

def main():
    print("Loading datasets...")
    # Update these paths if necessary based on your current fold
    gold_file = "data/5_fold_exp/train.spacy"
    silver_file = "data/pseudo_labeled_training_corpus.spacy"
    llm_file = "data/synthetic_pseudo_labeled_few_shot_v3.spacy"

    gold_texts = load_sentences_from_spacy(gold_file)
    silver_texts = load_sentences_from_spacy(silver_file)
    llm_texts = load_sentences_from_spacy(llm_file)
    
    # Subsample to avoid memory crash and overlapping blobs (adjust as your RAM allows)
    SAMPLE_SIZE = 2000 
    gold_texts = np.random.choice(gold_texts, min(SAMPLE_SIZE, len(gold_texts)), replace=False)
    silver_texts = np.random.choice(silver_texts, min(SAMPLE_SIZE, len(silver_texts)), replace=False)
    llm_texts = np.random.choice(llm_texts, min(SAMPLE_SIZE, len(llm_texts)), replace=False)

    print(f"Extracted {len(gold_texts)} Gold, {len(silver_texts)} Silver, and {len(llm_texts)} LLM sentences.")

    print("Generating MiniLM embeddings (this may take a minute)...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    all_texts = list(gold_texts) + list(silver_texts) + list(llm_texts)
    embeddings = embedder.encode(all_texts, show_progress_bar=True)

    print("Fitting t-SNE algorithm to map high-dimensional embeddings to 2D space...")
    # perplexity controls the balance between local and global aspects of your data
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    embeddings_2d = tsne.fit_transform(embeddings)

    # Split the 2D coordinates back into their respective datasets
    len_g = len(gold_texts)
    len_s = len(silver_texts)
    
    gold_2d = embeddings_2d[:len_g]
    silver_2d = embeddings_2d[len_g:len_g + len_s]
    llm_2d = embeddings_2d[len_g + len_s:]

    # Create Output Directory
    os.makedirs("visualizations", exist_ok=True)
    
    # Global Plot Settings
    alpha_val = 0.6
    dot_size = 15

    print("Plotting individual and combined graphs...")

    # --- PLOT 1: ALL DATASETS COMBINED ---
    plt.figure(figsize=(10, 8))
    plt.scatter(gold_2d[:, 0], gold_2d[:, 1], c='blue', label='Gold-Standard (train.spacy)', alpha=alpha_val, s=dot_size)
    plt.scatter(silver_2d[:, 0], silver_2d[:, 1], c='green', label='Silver-Standard (Real Corpus)', alpha=alpha_val, s=dot_size)
    plt.scatter(llm_2d[:, 0], llm_2d[:, 1], c='red', label='LLM-Generated (Synthetic)', alpha=alpha_val, s=dot_size)
    plt.title('t-SNE Overlay: Global Semantic Distribution')
    plt.legend()
    plt.savefig('visualizations/tsne_1_combined.png', dpi=300)
    plt.close()

    # --- PLOT 2: GOLD-STANDARD ONLY ---
    plt.figure(figsize=(10, 8))
    plt.scatter(gold_2d[:, 0], gold_2d[:, 1], c='blue', alpha=alpha_val, s=dot_size)
    # Fix the axes limits to match the combined graph so visual comparison is accurate
    plt.xlim(embeddings_2d[:, 0].min() - 5, embeddings_2d[:, 0].max() + 5)
    plt.ylim(embeddings_2d[:, 1].min() - 5, embeddings_2d[:, 1].max() + 5)
    plt.title('t-SNE: Internal Clusters of Gold-Standard Data')
    plt.savefig('visualizations/tsne_2_gold.png', dpi=300)
    plt.close()

    # --- PLOT 3: SILVER-STANDARD ONLY ---
    plt.figure(figsize=(10, 8))
    plt.scatter(silver_2d[:, 0], silver_2d[:, 1], c='green', alpha=alpha_val, s=dot_size)
    plt.xlim(embeddings_2d[:, 0].min() - 5, embeddings_2d[:, 0].max() + 5)
    plt.ylim(embeddings_2d[:, 1].min() - 5, embeddings_2d[:, 1].max() + 5)
    plt.title('t-SNE: Internal Clusters of Silver-Standard Data')
    plt.savefig('visualizations/tsne_3_silver.png', dpi=300)
    plt.close()

    # --- PLOT 4: LLM-GENERATED ONLY ---
    plt.figure(figsize=(10, 8))
    plt.scatter(llm_2d[:, 0], llm_2d[:, 1], c='red', alpha=alpha_val, s=dot_size)
    plt.xlim(embeddings_2d[:, 0].min() - 5, embeddings_2d[:, 0].max() + 5)
    plt.ylim(embeddings_2d[:, 1].min() - 5, embeddings_2d[:, 1].max() + 5)
    plt.title('t-SNE: Internal Clusters of LLM-Generated Data')
    plt.savefig('visualizations/tsne_4_llm.png', dpi=300)
    plt.close()

    print("[SUCCESS] All 4 t-SNE graphs saved to the 'visualizations' folder.")

if __name__ == "__main__":
    main()
