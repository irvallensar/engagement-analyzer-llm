import spacy
from spacy.tokens import DocBin
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def evaluate_aligned_with_o_tags(model_dir_pattern, test_dir_pattern, model_name):
    macro_f1_scores = []
    weighted_f1_scores = []
    
    print(f"\n==================================================")
    print(f"EVALUATING: {model_name} (CHAR-ALIGNED + 'O')")
    print(f"==================================================")
    
    for fold in range(1, 6):
        model_path = model_dir_pattern.format(fold)
        test_data_path = test_dir_pattern.format(fold)
        
        try:
            nlp = spacy.load(model_path)
            doc_bin = DocBin().from_disk(test_data_path)
            gold_docs = list(doc_bin.get_docs(nlp.vocab))
            
            y_true = []
            y_pred = []
            
            # Let RoBERTa read the raw text naturally so embeddings work perfectly
            print(f"  Running natural inference for Fold {fold}...")
            pred_docs = list(nlp.pipe([d.text for d in gold_docs]))
            
            for gold_doc, pred_doc in zip(gold_docs, pred_docs):
                doc_len = len(gold_doc)
                g_toks = [set() for _ in range(doc_len)]
                p_toks = [set() for _ in range(doc_len)]
                
                # 1. Map Gold Spans to Tokens
                if "sc" in gold_doc.spans:
                    for span in gold_doc.spans["sc"]:
                        for i in range(span.start, span.end):
                            g_toks[i].add(span.label_)
                            
                # 2. Map Predicted Spans via Character Alignment (Bypasses token mismatches)
                if "sc" in pred_doc.spans:
                    for p_span in pred_doc.spans["sc"]:
                        start_c = p_span.start_char
                        end_c = p_span.end_char
                        
                        # Find which gold tokens overlap with these characters
                        for token in gold_doc:
                            tok_start = token.idx
                            tok_end = token.idx + len(token)
                            
                            # If characters overlap, assign the label
                            if max(tok_start, start_c) < min(tok_end, end_c):
                                p_toks[token.i].add(p_span.label_)
                                
                # 3. Inject 'O' Tags
                for i in range(doc_len):
                    if not g_toks[i]: g_toks[i].add("O")
                    if not p_toks[i]: p_toks[i].add("O")
                    
                y_true.extend([list(s) for s in g_toks])
                y_pred.extend([list(s) for s in p_toks])
                
            # 4. Multi-Label Classification Math
            mlb = MultiLabelBinarizer()
            y_true_bin = mlb.fit_transform(y_true)
            y_pred_bin = mlb.transform(y_pred)
            
            report = classification_report(y_true_bin, y_pred_bin, target_names=mlb.classes_, output_dict=True, zero_division=0)
            
            # Print full table for fold 1 just so you can verify the 'O' tag math
            if fold == 1:
                print("\n--- FOLD 1 DETAILED BREAKDOWN ---")
                print(classification_report(y_true_bin, y_pred_bin, target_names=mlb.classes_, zero_division=0))
            
            fold_macro = report["macro avg"]["f1-score"]
            fold_weighted = report["weighted avg"]["f1-score"]
            fold_micro = report["micro avg"]["f1-score"]
            
            macro_f1_scores.append(fold_macro)
            weighted_f1_scores.append(fold_weighted)
            
            print(f"  -> Fold {fold} | Micro: {fold_micro:.4f} | Macro (with 'O'): {fold_macro:.4f} | Weighted: {fold_weighted:.4f}")
            
        except Exception as e:
            print(f"  [ERROR] Could not process Fold {fold}: {e}")

    final_macro = np.mean(macro_f1_scores) if macro_f1_scores else 0
    final_weighted = np.mean(weighted_f1_scores) if weighted_f1_scores else 0
    
    print("\n-----------------------------------------------------------")
    print(f"FINAL 5-FOLD AVERAGE -> Macro F1: {final_macro:.4f} | Weighted F1: {final_weighted:.4f}")
    print("-----------------------------------------------------------\n")


if __name__ == "__main__":
    TEST_DATA_PATTERN = "data/5_fold_exp/test{}.spacy"
    
    # Evaluate Baseline
    evaluate_aligned_with_o_tags("models/fold{}_baseline/model-best", TEST_DATA_PATTERN, "RoBERTa-base Baseline")
    
    # Evaluate Zero-Shot DA-RoBERTa
    evaluate_aligned_with_o_tags("models/fold{}_zero_shot/model-best", TEST_DATA_PATTERN, "Zero-Shot DA-RoBERTa")
    
    # Evaluate Few-Shot DA-RoBERTa
    evaluate_aligned_with_o_tags("models/fold{}_few_shot_v3/model-best", TEST_DATA_PATTERN, "Few-Shot DA-RoBERTa")
