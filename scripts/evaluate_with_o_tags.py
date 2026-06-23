import spacy
from spacy.tokens import DocBin, Doc
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Trap 2 Fix: Explicitly define all 11 classes so 'O' is never dropped by scikit-learn
TARGET_LABELS = [
    'ATTRIBUTION', 'CITATION', 'COUNTER', 'DENY', 
    'ENDOPHORIC', 'ENTERTAIN', 'JUSTIFYING', 
    'MONOGLOSS', 'O', 'PROCLAIM', 'SOURCES'
]

def evaluate_multi_label_with_o(model_dir_pattern, test_dir_pattern, model_name):
    macro_f1_scores = []
    weighted_f1_scores = []
    
    print(f"\n==================================================")
    print(f"EVALUATING: {model_name} (STRICT 11-CLASS MULTI-LABEL)")
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
            
            # FIX FOR THE 1.000 BUG: Create completely fresh, blank documents.
            # This strips all gold annotations before inference so the model cannot cheat.
            clean_docs = []
            for doc in gold_docs:
                words = [t.text for t in doc]
                spaces = [bool(t.whitespace_) for t in doc]
                clean_docs.append(Doc(nlp.vocab, words=words, spaces=spaces))
            
            print(f"  Running inference for Fold {fold}...")
            pred_docs = list(nlp.pipe(clean_docs))
            
            for gold_doc, pred_doc in zip(gold_docs, pred_docs):
                doc_len = len(gold_doc)
                
                # Arrays of sets for each token
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
                            
                # Inject 'O' Tags for empty tokens ONLY
                for i in range(doc_len):
                    if not g_toks[i]: 
                        g_toks[i].add("O")
                    if not p_toks[i]: 
                        p_toks[i].add("O")
                    
                y_true.extend([list(s) for s in g_toks])
                y_pred.extend([list(s) for s in p_toks])
                
            # Trap 3 Fix: Multi-Label Binarization (11-column binary matrix per token)
            mlb = MultiLabelBinarizer(classes=TARGET_LABELS)
            y_true_bin = mlb.fit_transform(y_true)
            y_pred_bin = mlb.transform(y_pred)
            
            # Generate the detailed report
            report_dict = classification_report(y_true_bin, y_pred_bin, target_names=TARGET_LABELS, output_dict=True, zero_division=0)
            report_text = classification_report(y_true_bin, y_pred_bin, target_names=TARGET_LABELS, zero_division=0)
            
            print(f"\n{'='*20} FOLD {fold} DETAILS {'='*20}")
            print(report_text)
            
            fold_macro = report_dict["macro avg"]["f1-score"]
            fold_weighted = report_dict["weighted avg"]["f1-score"]
            
            macro_f1_scores.append(fold_macro)
            weighted_f1_scores.append(fold_weighted)
            
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
    evaluate_multi_label_with_o("models/fold{}_baseline/model-best", TEST_DATA_PATTERN, "RoBERTa-base Baseline")
    
    # Evaluate DA-RoBERTa
    evaluate_multi_label_with_o("models/fold{}_zero_shot/model-best", TEST_DATA_PATTERN, "Zero-Shot DA-RoBERTa")
    evaluate_multi_label_with_o("models/fold{}_few_shot_v3/model-best", TEST_DATA_PATTERN, "Few-Shot DA-RoBERTa")
