import spacy
from spacy.tokens import DocBin
from sklearn.metrics import classification_report
import numpy as np
import warnings

# Suppress sklearn undefined metric warnings for clean output
warnings.filterwarnings("ignore")

def evaluate_model_with_o_tags(model_dir_pattern, test_dir_pattern):
    """
    Evaluates all 5 folds of a sequence tagger at the token level, 
    explicitly including 'O' (Outside) tags for background text.
    """
    macro_f1_scores = []
    weighted_f1_scores = []
    
    for fold in range(1, 6):
        model_path = model_dir_pattern.format(fold)
        test_data_path = test_dir_pattern.format(fold)
        
        try:
            print(f"Loading Fold {fold} from {model_path}...")
            nlp = spacy.load(model_path)
            doc_bin = DocBin().from_disk(test_data_path)
            docs = list(doc_bin.get_docs(nlp.vocab))
            
            y_true = []
            y_pred = []
            
            for doc in docs:
                # 1. Initialize all tokens in the document as "O" (Background)
                gold_labels = ["O"] * len(doc)
                pred_labels = ["O"] * len(doc)
                
                # 2. Map the Gold Spans (Overwrites "O" with the Engagement Label)
                if "sc" in doc.spans:
                    for span in doc.spans["sc"]:
                        for i in range(span.start, span.end):
                            gold_labels[i] = span.label_
                y_true.extend(gold_labels)
                
                # 3. Run Inference & Map Predicted Spans (Overwrites "O" with Prediction)
                pred_doc = nlp(doc.text)
                if "sc" in pred_doc.spans:
                    for span in pred_doc.spans["sc"]:
                        for i in range(span.start, span.end):
                            pred_labels[i] = span.label_
                y_pred.extend(pred_labels)
                
            # 4. Calculate Token-Level Metrics using sklearn
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            fold_macro = report["macro avg"]["f1-score"]
            fold_weighted = report["weighted avg"]["f1-score"]
            
            macro_f1_scores.append(fold_macro)
            weighted_f1_scores.append(fold_weighted)
            
            print(f"  -> Fold {fold} Macro F1 (with 'O'): {fold_macro:.4f}")
            
        except Exception as e:
            print(f"  [ERROR] Could not process Fold {fold}: {e}")

    # Calculate final 5-fold averages
    final_macro = np.mean(macro_f1_scores) if macro_f1_scores else 0
    final_weighted = np.mean(weighted_f1_scores) if weighted_f1_scores else 0
    
    return final_macro, final_weighted

if __name__ == "__main__":
    # --- CHECK THIS PATH TO MATCH YOUR BASELINE FOLDERS ---
    BASELINE_MODEL_PATTERN = "models/fold{}_baseline/model-best"
    TEST_DATA_PATTERN = "data/5_fold_exp/test{}.spacy"
    
    print("==================================================")
    print("EVALUATING RoBERTa-base BASELINE (INCLUDING 'O' TAGS)")
    print("==================================================")
    base_macro, base_weighted = evaluate_model_with_o_tags(BASELINE_MODEL_PATTERN, TEST_DATA_PATTERN)
    
    print("\n\nFINAL 5-FOLD AVERAGES (APPLES-TO-APPLES WITH EGUCHI & KYLE)")
    print("-----------------------------------------------------------")
    print(f"RoBERTa-base Baseline -> Macro F1: {base_macro:.4f} | Weighted F1: {base_weighted:.4f}")
