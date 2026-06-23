import spacy
from spacy.tokens import DocBin, Doc
from sklearn.metrics import classification_report
import numpy as np
import warnings

# Suppress sklearn undefined metric warnings for clean output
warnings.filterwarnings("ignore")

def evaluate_flattened_with_o_tags(model_dir_pattern, test_dir_pattern, model_name):
    macro_f1_scores = []
    weighted_f1_scores = []
    
    print(f"\n==================================================")
    print(f"EVALUATING: {model_name} (FLATTENED 1D + 'O' TAGS)")
    print(f"==================================================")
    
    for fold in range(1, 6):
        model_path = model_dir_pattern.format(fold)
        test_data_path = test_dir_pattern.format(fold)
        
        try:
            # Load the model and test data
            nlp = spacy.load(model_path)
            doc_bin = DocBin().from_disk(test_data_path)
            docs = list(doc_bin.get_docs(nlp.vocab))
            
            y_true = []
            y_pred = []
            
            # 1. Safely recreate documents to perfectly preserve your custom tokenization
            clean_docs = []
            for doc in docs:
                words = [t.text for t in doc]
                spaces = [bool(t.whitespace_) for t in doc]
                clean_docs.append(Doc(nlp.vocab, words=words, spaces=spaces))
            
            # 2. Run blazing fast batched inference
            print(f"  Running fast batched inference for Fold {fold}...")
            pred_docs = list(nlp.pipe(clean_docs))
            
            # 3. 1D Array Flattening (This mimics the Eguchi & Kyle Bi-LSTM math)
            for gold_doc, pred_doc in zip(docs, pred_docs):
                
                # Flatten Gold Spans
                g_labels = ["O"] * len(gold_doc)
                if "sc" in gold_doc.spans:
                    for span in gold_doc.spans["sc"]:
                        for i in range(span.start, span.end):
                            g_labels[i] = span.label_
                y_true.extend(g_labels)
                
                # Flatten Predicted Spans
                p_labels = ["O"] * len(pred_doc)
                if "sc" in pred_doc.spans:
                    for span in pred_doc.spans["sc"]:
                        for i in range(span.start, span.end):
                            p_labels[i] = span.label_
                y_pred.extend(p_labels)
                
            # 4. Calculate exactly like a traditional flattened classifier
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            fold_macro = report["macro avg"]["f1-score"]
            fold_weighted = report["weighted avg"]["f1-score"]
            
            macro_f1_scores.append(fold_macro)
            weighted_f1_scores.append(fold_weighted)
            
            print(f"  -> Fold {fold} Macro F1 (with 'O'): {fold_macro:.4f} | Weighted: {fold_weighted:.4f}")
            
        except Exception as e:
            print(f"  [ERROR] Could not process Fold {fold}: {e}")

    # Calculate final 5-fold averages
    final_macro = np.mean(macro_f1_scores) if macro_f1_scores else 0
    final_weighted = np.mean(weighted_f1_scores) if weighted_f1_scores else 0
    
    print("-----------------------------------------------------------")
    print(f"FINAL 5-FOLD AVERAGE -> Macro F1: {final_macro:.4f} | Weighted F1: {final_weighted:.4f}")
    print("-----------------------------------------------------------\n")


if __name__ == "__main__":
    TEST_DATA_PATTERN = "data/5_fold_exp/test{}.spacy"
    
    # 1. Evaluate Baseline
    evaluate_flattened_with_o_tags("models/fold{}_baseline/model-best", TEST_DATA_PATTERN, "RoBERTa-base Baseline")
    
    # 2. Evaluate Zero-Shot
    evaluate_flattened_with_o_tags("models/fold{}_zero_shot/model-best", TEST_DATA_PATTERN, "Zero-Shot DA-RoBERTa")
    
    # 3. Evaluate Few-Shot
    evaluate_flattened_with_o_tags("models/fold{}_few_shot_v3/model-best", TEST_DATA_PATTERN, "Few-Shot DA-RoBERTa")
