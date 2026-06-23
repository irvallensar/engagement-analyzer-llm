import spacy
from spacy.tokens import DocBin, Doc
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import warnings

# Suppress sklearn undefined metric warnings for clean output
warnings.filterwarnings("ignore")

def evaluate_multilabel_with_o_tags(model_dir_pattern, test_dir_pattern, model_name):
    macro_f1_scores = []
    weighted_f1_scores = []
    
    print(f"\n==================================================")
    print(f"EVALUATING: {model_name} (TOKEN-LEVEL MULTI-LABEL + 'O')")
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
            
            # 2. Run fast batched inference
            print(f"  Running batched inference for Fold {fold}...")
            pred_docs = list(nlp.pipe(clean_docs))
            
            # 3. Independent Multi-Layer Token Mapping (Solves Overlaps)
            for gold_doc, pred_doc in zip(docs, pred_docs):
                doc_len = len(gold_doc)
                
                # Each token gets a SET of labels
                g_toks = [set() for _ in range(doc_len)]
                p_toks = [set() for _ in range(doc_len)]
                
                # Map Gold Spans
                if "sc" in gold_doc.spans:
                    for span in gold_doc.spans["sc"]:
                        for i in range(span.start, span.end):
                            g_toks[i].add(span.label_)
                            
                # Map Predicted Spans
                if "sc" in pred_doc.spans:
                    for span in pred_doc.spans["sc"]:
                        for i in range(span.start, span.end):
                            p_toks[i].add(span.label_)
                            
                # Assign 'O' ONLY if the token has no other labels
                for i in range(doc_len):
                    if not g_toks[i]: g_toks[i].add("O")
                    if not p_toks[i]: p_toks[i].add("O")
                    
                # Convert sets to lists and append to master arrays
                y_true.extend([list(s) for s in g_toks])
                y_pred.extend([list(s) for s in p_toks])
                
            # 4. Multi-Label Binarization (Creates independent layers per category)
            mlb = MultiLabelBinarizer()
            y_true_bin = mlb.fit_transform(y_true)
            y_pred_bin = mlb.transform(y_pred)
            
            # 5. Calculate Metrics
            report = classification_report(y_true_bin, y_pred_bin, target_names=mlb.classes_, output_dict=True, zero_division=0)
            # --- ADD THIS LINE TO PRINT THE FULL TABLE ---
            print(classification_report(y_true_bin, y_pred_bin, target_names=mlb.classes_, zero_division=0))
            # ---------------------------------------------
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
    evaluate_multilabel_with_o_tags("models/fold{}_baseline/model-best", TEST_DATA_PATTERN, "RoBERTa-base Baseline")
    
    # 2. Evaluate Zero-Shot DA-RoBERTa
    evaluate_multilabel_with_o_tags("models/fold{}_zero_shot/model-best", TEST_DATA_PATTERN, "Zero-Shot DA-RoBERTa")
    
    # 3. Evaluate Few-Shot DA-RoBERTa
    evaluate_multilabel_with_o_tags("models/fold{}_few_shot_v3/model-best", TEST_DATA_PATTERN, "Few-Shot DA-RoBERTa")
