import spacy
from spacy.tokens import DocBin
from spacy.training.example import Example
from collections import defaultdict
import os

# =====================================================================
# EDIT THESE PATHS TO MATCH YOUR EXACT DIRECTORY NAMES
# Use '{}' where the fold number (1-5) should go.
# =====================================================================
MODELS = {
    "RoBERTa-base (Baseline)": "./models/fold{}/model-best", 
    "DA-RoBERTa (Zero-Shot)": "./models/fold{}_zero_shot/model-best",
    "DA-RoBERTa (Few-Shot V3)": "./models/fold{}_few_shot_v3/model-best"
}

TEST_DATA_PATH = "./data/5_fold_exp/test{}.spacy"
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

def get_gold_support(ref_docs):
    """Calculates the exact count of each label in the test set for Weighted F1."""
    support = defaultdict(int)
    total_spans = 0
    for doc in ref_docs:
        for span in doc.spans.get("sc", []):
            support[span.label_] += 1
            total_spans += 1
    return support, total_spans

def main():
    print("============================================================")
    print("   THRESHOLD (5-FOLD CV | MACRO & WEIGHTED F1)   ")
    print("============================================================\n")

    for model_name, path_template in MODELS.items():
        print(f"Evaluating: {model_name}")
        print("-" * 65)
        print(f"{'Threshold':<10} | {'Macro F1':<10} | {'Weighted F1':<12} | {'Precision':<10} | {'Recall':<10}")
        print("-" * 65)

        # To hold the accumulated metrics across all 5 folds for each threshold
        threshold_metrics = {t: {"macro_f1": 0, "weighted_f1": 0, "p": 0, "r": 0} for t in THRESHOLDS}
        valid_folds = 0

        for fold in range(1, 6):
            model_path = path_template.format(fold)
            test_path = TEST_DATA_PATH.format(fold)

            if not os.path.exists(model_path):
                print(f"  [Warning] Missing model fold: {model_path}")
                continue
                
            valid_folds += 1
            nlp = spacy.load(model_path)
            doc_bin = DocBin().from_disk(test_path)
            ref_docs = list(doc_bin.get_docs(nlp.vocab))
            
            # Extract support for Weighted Math
            support_dict, total_spans = get_gold_support(ref_docs)

            for t in THRESHOLDS:
                nlp.get_pipe("spancat").cfg["threshold"] = t

                examples = []
                for doc in ref_docs:
                    predicted = nlp.make_doc(doc.text)
                    predicted = nlp(predicted)
                    examples.append(Example(predicted, doc))

                scores = nlp.evaluate(examples)
                
                # Global metrics
                macro_f1 = scores.get("spans_sc_f", 0.0) * 100
                p = scores.get("spans_sc_p", 0.0) * 100
                r = scores.get("spans_sc_r", 0.0) * 100
                
                # Per-label metrics to calculate Weighted F1
                per_type_scores = scores.get("spans_sc_per_type", {})
                weighted_f1_sum = 0.0
                
                for label, label_scores in per_type_scores.items():
                    label_f1 = label_scores.get("f", 0.0) * 100
                    label_support = support_dict.get(label, 0)
                    weighted_f1_sum += (label_f1 * label_support)
                
                weighted_f1 = weighted_f1_sum / total_spans if total_spans > 0 else 0.0

                # Accumulate for averaging later
                threshold_metrics[t]["macro_f1"] += macro_f1
                threshold_metrics[t]["weighted_f1"] += weighted_f1
                threshold_metrics[t]["p"] += p
                threshold_metrics[t]["r"] += r

        # Calculate and print the 5-Fold Averages for this model
        if valid_folds > 0:
            for t in THRESHOLDS:
                avg_macro = threshold_metrics[t]["macro_f1"] / valid_folds
                avg_weight = threshold_metrics[t]["weighted_f1"] / valid_folds
                avg_p = threshold_metrics[t]["p"] / valid_folds
                avg_r = threshold_metrics[t]["r"] / valid_folds
                
                print(f"{t:<10.1f} | {avg_macro:<10.2f} | {avg_weight:<12.2f} | {avg_p:<10.2f} | {avg_r:<10.2f}")
        else:
            print(f"  -> Could not evaluate {model_name} (Models not found).")
            
        print("============================================================\n")

if __name__ == "__main__":
    main()
