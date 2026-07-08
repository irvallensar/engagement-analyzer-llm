import spacy
from spacy.tokens import DocBin
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch

def load_texts_from_spacy(file_path):
    print(f"Reading texts from {file_path}...")

    # Create a lightweight SpaCy pipeline.
    # We only need the vocabulary to reconstruct Doc objects from the DocBin file,
    # so loading a full model (e.g. en_core_web_sm) would be unnecessary overhead.
    nlp = spacy.blank("en")

    # Load the serialized SpaCy documents from disk.
    doc_bin = DocBin().from_disk(file_path)

    # Reconstruct individual Doc objects from the binary container.
    docs = list(doc_bin.get_docs(nlp.vocab))

    # Empty or whitespace-only documents are ignored so they do not affect embedding quality metrics.
    return [doc.text for doc in docs if len(doc.text.strip()) > 0]

def main():
    # STEP 1: Load both datasets
    # The human dataset serves as the reference corpus.
    # The synthetic dataset will be compared against it.
    human_texts = load_texts_from_spacy("data/train.spacy")
    synthetic_texts = load_texts_from_spacy("data/pseudo_labeled_training_corpus.spacy")

    print(f"Loaded {len(human_texts)} human sentences.")
    print(f"Loaded {len(synthetic_texts)} synthetic sentences.")
    
    # STEP 2: Initialize embedding model
    print("\nLoading sentence-transformers/all-MiniLM-L6-v2...")

    # Hardware Selection:
    # 1. CUDA GPU (NVIDIA)
    # 2. MPS (Apple Silicon)
    # 3. CPU fallback
    #
    # This allows the script to run efficiently across different hardware.
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    # Load a compact Sentence Transformer model that generates
    # semantically meaningful sentence embeddings.
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    print(f"Running on device: {device}")
    
    # STEP 3: Generate embeddings
    # Each sentence is converted into a dense vector representation.
    # Similar meanings should produce vectors that are close together.
    print("Embedding human sentences (this may take a minute)...")
    human_embeddings = model.encode(
        human_texts,
        convert_to_tensor=True,
        show_progress_bar=True
    )
    print("Embedding synthetic sentences...")
    synthetic_embeddings = model.encode(
        synthetic_texts,
        convert_to_tensor=True,
        show_progress_bar=True
    )
    
    # STEP 4: Measure semantic similarity
    print("\nCalculating semantic similarity matrix...")
    
    # Computing a full similarity matrix can consume a large amount
    # of memory when datasets become large.
    #
    # Example:
    # 10,000 synthetic x 50,000 human sentences
    # would produce a matrix with 500 million similarity scores.
    #
    # To avoid memory issues, process synthetic embeddings in chunks.
    batch_size = 1000
    all_max_similarities = []    
    # Iterate through synthetic embeddings batch-by-batch.
    # Each iteration computes similarities for only a subset
    # of the synthetic corpus.
    for i in range(0, len(synthetic_embeddings), batch_size):
        # Current slice of synthetic embeddings.
        batch_syn = synthetic_embeddings[i:i+batch_size]
        # Cosine similarity compares semantic direction rather than magnitude.
        # Higher values indicate stronger semantic similarity.
        similarity_matrix = util.cos_sim(batch_syn, human_embeddings)        
        # For each synthetic sentence:
        # Find its closest matching human sentence.

        # dim=1 means:
        # "take the maximum value across all human sentences
        # for every synthetic sentence."
        max_sim_per_sentence = torch.max(similarity_matrix, dim=1).values
        # Move results from GPU/MPS to CPU and convert to Python numbers.
        all_max_similarities.extend(max_sim_per_sentence.cpu().tolist())
    # Convert to NumPy array so statistical operations can be
    # computed efficiently.
    all_max_similarities = np.array(all_max_similarities)
    

    # STEP 5: Statistical summary
    # Mean:
    # Average similarity between each synthetic sentence and
    # its closest human counterpart.
    mean_sim = np.mean(all_max_similarities)

    # Standard deviation:
    # Indicates how spread out similarity scores are.
    # Large values suggest inconsistent similarity levels.
    std_sim = np.std(all_max_similarities)

    # Extreme values provide insight into best- and worst-case matches.
    min_sim = np.min(all_max_similarities)
    max_sim = np.max(all_max_similarities)

    # Distributional proportions:
    # These thresholds give an interpretable breakdown of where
    # the similarity scores fall, rather than relying on mean/SD alone.
    # Distribution across non-overlapping similarity ranges.
    # These four buckets cover 100% of all similarity scores.

    prop_lt_03 = np.mean(all_max_similarities < 0.3)

    prop_03_05 = np.mean(
        (all_max_similarities >= 0.3) &
        (all_max_similarities < 0.5)
    )

    prop_05_07 = np.mean(
        (all_max_similarities >= 0.5) &
        (all_max_similarities <= 0.7)
    )

    prop_gt_07 = np.mean(all_max_similarities > 0.7)

    print("\n" + "="*40)
    print("       DATA QUALITY METRIC RESULTS")
    print("="*40)
    print(f"Mean Cosine Similarity: {mean_sim:.4f}")
    print(f"Standard Deviation:     {std_sim:.4f}")
    print(f"Minimum Similarity:     {min_sim:.4f}")
    print(f"Maximum Similarity:     {max_sim:.4f}")
    print("-"*40)
    print("  DISTRIBUTION BY SIMILARITY RANGE")
    print("-"*40)
    print(f"< 0.3:       {prop_lt_03:.2%}")
    print(f"0.3 - 0.5:   {prop_03_05:.2%}")
    print(f"0.5 - 0.7:   {prop_05_07:.2%}")
    print(f"> 0.7:       {prop_gt_07:.2%}")

    total_prop = (
        prop_lt_03 +
        prop_03_05 +
        prop_05_07 +
        prop_gt_07
    )

    print("-"*40)
    print(f"Coverage Check: {total_prop:.2%}")
    print("="*40)
    
    # STEP 6: Export results
    # Persist metrics so they can be reviewed later or included
    # in experiment reports without rerunning the script.
    output_path = "data/synthetic_similarity_report.txt"

    # Context manager automatically closes the file even if an
    # exception occurs while writing.
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== Sentence Transformer Similarity Report ===\n")
        f.write(f"Model Used: sentence-transformers/all-MiniLM-L6-v2\n")
        f.write(f"Human Dataset Size: {len(human_texts)} sentences\n")
        f.write(f"Synthetic Dataset Size: {len(synthetic_texts)} sentences\n\n")
        f.write(f"Mean Cosine Similarity: {mean_sim:.4f}\n")
        f.write(f"Standard Deviation: {std_sim:.4f}\n")
        f.write(f"Min Similarity: {min_sim:.4f}\n")
        f.write(f"Max Similarity: {max_sim:.4f}\n")
        f.write("\nDistribution by Similarity Range:\n")
        f.write(f"< 0.3:       {prop_lt_03:.2%}\n")
        f.write(f"0.3 - 0.5:   {prop_03_05:.2%}\n")
        f.write(f"0.5 - 0.7:   {prop_05_07:.2%}\n")
        f.write(f"> 0.7:       {prop_gt_07:.2%}\n")

    print(f"[SUCCESS] Metrics exported to {output_path}")

if __name__ == "__main__":
    main()
