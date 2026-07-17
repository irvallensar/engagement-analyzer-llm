import spacy
from spacy.tokens import DocBin
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_sentences(file_path):
    nlp = spacy.blank("en")
    doc_bin = DocBin().from_disk(file_path)
    return [doc.text for doc in doc_bin.get_docs(nlp.vocab)]

def calculate_internal_similarity(texts, embedder, sample_size=2000):
    # Subsample to keep memory usage safe and computation fast
    # 2000 sentences yields roughly 2 million unique pairs
    if len(texts) > sample_size:
        texts = np.random.choice(texts, sample_size, replace=False)
    
    print(f"Encoding {len(texts)} sentences...")
    embeddings = embedder.encode(texts, show_progress_bar=True)
    
    print("Calculating pairwise cosine similarity matrix...")
    sim_matrix = cosine_similarity(embeddings)
    
    # Extract the upper triangle of the matrix 
    # (This gets all unique pairs and excludes the 1.0 diagonal where sentences match themselves)
    upper_triangle_indices = np.triu_indices_from(sim_matrix, k=1)
    pairwise_similarities = sim_matrix[upper_triangle_indices]
    
    mean_sim = np.mean(pairwise_similarities)
    std_sim = np.std(pairwise_similarities)
    
    return mean_sim, std_sim

def main():
    # File paths
    gold_file = "data/train.spacy"
    silver_file = "data/pseudo_labeled_training_corpus.spacy"
    llm_file = "data/synthetic_pseudo_labeled_few_shot_v3.spacy"

    print("Loading datasets...")
    gold_texts = load_sentences(gold_file)
    silver_texts = load_sentences(silver_file)
    llm_texts = load_sentences(llm_file)

    # Initialize MiniLM
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    print("\n========================================")
    print("   INTERNAL PAIRWISE SIMILARITY SCORES")
    print("========================================")

    print("\n--- Gold-Standard (Human) ---")
    g_mean, g_std = calculate_internal_similarity(gold_texts, embedder)
    print(f"Average Internal Similarity: {g_mean:.4f} (Std: {g_std:.4f})")

    print("\n--- Silver-Standard (Real Corpus) ---")
    s_mean, s_std = calculate_internal_similarity(silver_texts, embedder)
    print(f"Average Internal Similarity: {s_mean:.4f} (Std: {s_std:.4f})")

    print("\n--- LLM-Generated (Synthetic Few-Shot) ---")
    l_mean, l_std = calculate_internal_similarity(llm_texts, embedder)
    print(f"Average Internal Similarity: {l_mean:.4f} (Std: {l_std:.4f})")

if __name__ == "__main__":
    main()
