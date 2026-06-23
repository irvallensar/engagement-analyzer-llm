import spacy
from spacy.tokens import DocBin
from collections import defaultdict
import numpy as np

def evaluate_model_with_o_tags(model_dir_pattern, test_dir_pattern, model_name):
    """
    Evaluates all 5 folds using strict Set Math to accurately handle overlapping 
    spans and explicitly includes 'O' (Outside) tags for background text.
    """
    macro_f1_scores = []
    weighted_f1_scores = []
    
    print(f"\n==================================================")
    print(f"EVALUATING: {model_name} (INCLUDING 'O' TAGS)")
    print(f"==================================================")
    
    for fold in range(1, 6):
        model_path = model_dir_pattern.format(fold)
        test_data_path = test_dir_pattern.format(fold)
        
        try:
            print(f"Loading Fold {fold} from {model_path}...")
            nlp = spacy.load(model_path)
            doc_bin = DocBin().from_disk(test_data_path)
            docs = list(doc_bin.get_docs(nlp.vocab))
            
            cat_tp = defaultdict(int)
            cat_fp = defaultdict(int)
            cat_fn = defaultdict(int)
            
            for doc in docs:
                # 1. Map Gold Tokens
                gold_tokens = set()
                gold_has_label = set()
                
                if "sc" in doc.spans:
                    for span in doc.spans["sc"]:
                        for i in range(span.start, span.end):
                            gold_tokens.add((i, span.label_))
                            gold_has_label.add(i)
                            
                # Assign 'O' to tokens with no engagement label
                for i in range(len(doc)):
                    if i not in gold_has_label:
                        gold_tokens.add((i, "O"))
                        
                # 2. Predict Spans (Preserving original tokenization)
                pred_doc = doc.copy()
                pred_doc.spans.clear() # Clear gold spans before prediction
                
                # Pass the doc through the model pipeline
                for name, proc in nlp.pipeline:
                    pred_doc = proc(pred_doc)
                    
                # 3. Map Predicted Tokens
                pred_tokens = set()
                pred_has_label = set()
                
                if "sc" in pred_doc.spans:
                    for span in pred_doc.spans["sc"]:
                        for i in range(span.start, span.end):
                            pred_tokens.add((i, span.label_))
                            pred_has_label.add(i)
                            
                # Assign 'O' to predicted background tokens
                for i in range(len(pred_doc)):
                    if i not in pred_has_label:
                        pred_tokens.add((i, "O"))
                        
                # 4. Set Intersection Math (Handles Overlaps flawlessly)
                tok_tp = gold_tokens.intersection(pred_tokens)
                tok_fp = pred_tokens - gold_tokens
                tok_fn = gold_tokens - pred_tokens
                
                for idx, label in tok_tp: cat_tp[label] += 1
                for idx, label in tok_fp: cat_fp[label] += 1
                for idx, label in tok_fn: cat_fn[label] += 1

            # 5. Calculate Fold Averages
            all_labels = set(list(cat_tp.keys()) + list(cat_fp.keys()) + list(cat_fn.keys()))
            macro_f1_sum = 0
            weighted_f1_sum = 0
            total_support = 0
            
            for label in all_labels:
                tp = cat_tp[label]
                fp = cat_fp[label]
                fn = cat_fn[label]
                support = tp + fn
                total_support += support
                
                p = tp / (tp + fp) if (tp + fp) > 0 else 0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
                
                macro_f1_sum += f1
                weighted_f1_sum += f1 * support
                
            fold_macro = macro_f1_sum / len(all_labels) if len(all_labels) > 0 else 0
            fold_weighted = weighted_f1_sum / total_support if total_support > 0 else 0
            
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
    evaluate_model_with_o_tags("models/fold{}_baseline/model-best", TEST_DATA_PATTERN, "RoBERTa-base Baseline")
    
    # 2. Evaluate Zero-Shot
    evaluate_model_with_o_tags("models/fold{}_zero_shot/model-best", TEST_DATA_PATTERN, "Zero-Shot DA-RoBERTa")
    
    # 3. Evaluate Few-Shot
    evaluate_model_with_o_tags("models/fold{}_few_shot_v3/model-best", TEST_DATA_PATTERN, "Few-Shot DA-RoBERTa")
