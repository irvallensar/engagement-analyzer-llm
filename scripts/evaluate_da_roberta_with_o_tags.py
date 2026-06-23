import spacy
from spacy.tokens import DocBin
from sklearn.metrics import classification_report
import numpy as np
import warnings

# Suppress sklearn undefined metric warnings for clean output
warnings.filterwarnings("ignore")

def evaluate_model_with_o_tags(model_dir_pattern, test_dir_pattern):
    """
    Evaluates all 5 folds of a DA-RoBERTa model at the token level, 
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
                
            # 4. Calculate Token-Level Metrics using sklearn (which includes the massive "O" class)
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
    # --- EDIT THESE PATHS TO MATCH YOUR LOCAL DIRECTORIES ---
    # Use {} where the fold number should go.
    # Example: "models/fold{}_zero_shot/model-best"
    
    ZERO_SHOT_MODEL_PATTERN = "models/fold{}_zero_shot/model-best"
    FEW_SHOT_MODEL_PATTERN = "models/fold{}_few_shot_v3/model-best"
    TEST_DATA_PATTERN = "data/5_fold_exp/test{}.spacy"
    
    print("==================================================")
    print("EVALUATING ZERO-SHOT DA-RoBERTa (INCLUDING 'O' TAGS)")
    print("==================================================")
    zs_macro, zs_weighted = evaluate_model_with_o_tags(ZERO_SHOT_MODEL_PATTERN, TEST_DATA_PATTERN)
    
    print("\n==================================================")
    print("EVALUATING FEW-SHOT DA-RoBERTa (INCLUDING 'O' TAGS)")
    print("==================================================")
    fs_macro, fs_weighted = evaluate_model_with_o_tags(FEW_SHOT_MODEL_PATTERN, TEST_DATA_PATTERN)
    
    print("\n\nFINAL 5-FOLD AVERAGES (APPLES-TO-APPLES WITH EGUCHI & KYLE)")
    print("-----------------------------------------------------------")
    print(f"Zero-Shot DA-RoBERTa -> Macro F1: {zs_macro:.4f} | Weighted F1: {zs_weighted:.4f}")
    print(f"Few-Shot DA-RoBERTa  -> Macro F1: {fs_macro:.4f} | Weighted F1: {fs_weighted:.4f}")
