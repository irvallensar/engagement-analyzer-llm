import spacy
from spacy.tokens import DocBin
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch

def load_texts_from_spacy(file_path):
    print(f"Reading texts from {file_path}...")
    # A blank English pipeline is enough to unpack DocBin documents without running a full NLP model.
    nlp = spacy.blank("en")
    doc_bin = DocBin().from_disk(file_path)
    docs = list(doc_bin.get_docs(nlp.vocab))
    # Empty or whitespace-only documents are ignored so they do not affect embedding quality metrics.
    return [doc.text for doc in docs if len(doc.text.strip()) > 0]

def main():
    # 1. Load the text sets
    human_texts = load_texts_from_spacy("data/train.spacy")
    synthetic_texts = load_texts_from_spacy("data/synthetic_pseudo_labeled_zero_shot.spacy")
    
    print(f"Loaded {len(human_texts)} human sentences.")
    print(f"Loaded {len(synthetic_texts)} synthetic sentences.")
    
    # 2. Initialize the Sentence Transformer
    print("\nLoading sentence-transformers/all-MiniLM-L6-v2...")
    # Prefer GPU acceleration when available, then Apple Silicon MPS, and otherwise fall back to CPU.
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    print(f"Running on device: {device}")
    
    # 3. Compute Embeddings
    print("Embedding human sentences (this may take a minute)...")
    human_embeddings = model.encode(human_texts, convert_to_tensor=True, show_progress_bar=True)
    
    print("Embedding synthetic sentences...")
    synthetic_embeddings = model.encode(synthetic_texts, convert_to_tensor=True, show_progress_bar=True)
    
    # 4. Calculate Cosine Similarity
    print("\nCalculating semantic similarity matrix...")
    
    # To prevent out-of-memory errors on large corpuses, we process in batches
    batch_size = 1000
    all_max_similarities = []
    
    for i in range(0, len(synthetic_embeddings), batch_size):
        # Slice the synthetic embeddings so each similarity matrix stays within memory limits.
        batch_syn = synthetic_embeddings[i:i+batch_size]
        # Compute cosine similarity matrix for this batch against all human embeddings
        similarity_matrix = util.cos_sim(batch_syn, human_embeddings)
        
        # Find the highest similarity score for each synthetic sentence to ANY human sentence
        max_sim_per_sentence = torch.max(similarity_matrix, dim=1).values
        all_max_similarities.extend(max_sim_per_sentence.cpu().tolist())
    # Convert to NumPy so the summary statistics below can be computed directly.        
    all_max_similarities = np.array(all_max_similarities)
    
    # 5. Compile Statistical Summary
    mean_sim = np.mean(all_max_similarities)
    # These values summarize how closely synthetic sentences resemble the nearest human-written sentence.
    std_sim = np.std(all_max_similarities)
    min_sim = np.min(all_max_similarities)
    max_sim = np.max(all_max_similarities)
    
    print("\n" + "="*40)
    print("       DATA QUALITY METRIC RESULTS")
    print("="*40)
    print(f"Mean Cosine Similarity: {mean_sim:.4f}")
    print(f"Standard Deviation:     {std_sim:.4f}")
    print(f"Minimum Similarity:     {min_sim:.4f}")
    print(f"Maximum Similarity:     {max_sim:.4f}")
    print("="*40)
    
    # Save results to a file for your thesis appendix
    output_path = "data/synthetic_similarity_report.txt"
    # The report mirrors the console output and records dataset sizes for reproducibility.
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== Sentence Transformer Similarity Report ===\n")
        f.write(f"Model Used: sentence-transformers/all-MiniLM-L6-v2\n")
        f.write(f"Human Dataset Size: {len(human_texts)} sentences\n")
        f.write(f"Synthetic Dataset Size: {len(synthetic_texts)} sentences\n\n")
        f.write(f"Mean Cosine Similarity: {mean_sim:.4f}\n")
        f.write(f"Standard Deviation: {std_sim:.4f}\n")
        f.write(f"Min Similarity: {min_sim:.4f}\n")
        f.write(f"Max Similarity: {max_sim:.4f}\n")
    print(f"[SUCCESS] Metrics exported to {output_path}")

if __name__ == "__main__":
    main()
