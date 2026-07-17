import spacy
from spacy.tokens import DocBin
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_sentences_by_category(file_path, target_category, spans_key="sc"):
    nlp = spacy.blank("en")
    doc_bin = DocBin().from_disk(file_path)
    docs = list(doc_bin.get_docs(nlp.vocab))
    
    filtered_texts = []
    for doc in docs:
        if any(span.label_ == target_category for span in doc.spans.get(spans_key, [])):
            filtered_texts.append(doc.text)
    return filtered_texts

def calculate_internal_similarity(texts, embedder, sample_size=2000):
    # Adjusted sample size to 2000 to match the t-SNE analysis
    if len(texts) > sample_size:
        texts = np.random.choice(texts, sample_size, replace=False)
    
    embeddings = embedder.encode(texts, show_progress_bar=False)
    sim_matrix = cosine_similarity(embeddings)
    
    upper_triangle_indices = np.triu_indices_from(sim_matrix, k=1)
    pairwise_similarities = sim_matrix[upper_triangle_indices]
    
    return np.mean(pairwise_similarities), np.std(pairwise_similarities)

def main():
    TARGET_CATEGORY = "SOURCES"
    
    print(f"Loading datasets for category: {TARGET_CATEGORY}...")
    gold_file = "data/train.spacy"
    silver_file = "data/pseudo_labeled_training_corpus.spacy"
    llm_file = "data/synthetic_pseudo_labeled_few_shot_v3.spacy"

    gold_texts = load_sentences_by_category(gold_file, TARGET_CATEGORY)
    silver_texts = load_sentences_by_category(silver_file, TARGET_CATEGORY)
    llm_texts = load_sentences_by_category(llm_file, TARGET_CATEGORY)

    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    print(f"\nINTERNAL PAIRWISE SIMILARITY ({TARGET_CATEGORY})")
    print("--------------------------------------------------")
    
    g_mean, g_std = calculate_internal_similarity(gold_texts, embedder)
    print(f"Gold-Standard (Human):     {g_mean:.4f} (Std: {g_std:.4f})")

    s_mean, s_std = calculate_internal_similarity(silver_texts, embedder)
    print(f"Silver-Standard (Real):    {s_mean:.4f} (Std: {s_std:.4f})")

    l_mean, l_std = calculate_internal_similarity(llm_texts, embedder)
    print(f"LLM-Generated (Synthetic): {l_mean:.4f} (Std: {l_std:.4f})")

if __name__ == "__main__":
    main()
