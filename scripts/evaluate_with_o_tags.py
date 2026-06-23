import spacy
from spacy.tokens import DocBin
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def evaluate_pipeline(model_dir_pattern, test_dir_pattern, model_name, threshold=0.30):
    all_fold_reports = []
    
    print(f"\n==================================================")
    print(f"EVALUATING: {model_name}")
    print(f"SPAN CONFIDENCE THRESHOLD: {threshold}")
    print(f"==================================================")
    
    for fold in range(1, 6):
        model_path = model_dir_pattern.format(fold)
        test_data_path = test_dir_pattern.format(fold)
        
        try:
            nlp = spacy.load(model_path)
            
            # Tune the spancat threshold to fix the low recall
            if "spancat" in nlp.pipe_names:
                nlp.get_pipe("spancat").cfg["threshold"] = threshold
            elif "spancat_single" in nlp.pipe_names:
                nlp.get_pipe("spancat_single").cfg["threshold"] = threshold
                
            doc_bin = DocBin().from_disk(test_data_path)
            gold_docs = list(doc_bin.get_docs(nlp.vocab))
            
            y_true = []
            y_pred = []
            
            # Run inference using the original document tokens to preserve structure
            pred_docs = list(nlp.pipe(gold_docs))
            
            for gold_doc, pred_doc in zip(gold_docs, pred_docs):
                doc_len = len(gold_doc)
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
                            
                # Inject 'O' Tags for empty tokens
                for i in range(doc_len):
                    if not g_toks[i]: g_toks[i].add("O")
                    if not p_toks[i]: p_toks[i].add("O")
                    
                y_true.extend([list(s) for s in g_toks])
                y_pred.extend([list(s) for s in p_toks])
                
            # Process metrics
            mlb = MultiLabelBinarizer()
            y_true_bin = mlb.fit_transform(y_true)
            y_pred_bin = mlb.transform(y_pred)
            
            # Print the complete breakdown for this fold exactly like the .md log
            print(f"\n{"="*20} FOLD {fold} DETAILS {"="*20}")
            fold_report_text = classification_report(y_true_bin, y_pred_bin, target_names=mlb.classes_, zero_division=0)
            print(fold_report_text)
            
            fold_report_dict = classification_report(y_true_bin, y_pred_bin, target_names=mlb.classes_, output_dict=True, zero_division=0)
            all_fold_reports.append(fold_report_dict)
            
        except Exception as e:
            print(f"  [ERROR] Could not process Fold {fold}: {e}")

    # Calculate Cross-Validation Averages correctly across folds
    if all_fold_reports:
        macro_f1s = [r["macro avg"]["f1-score"] for r in all_fold_reports]
        weighted_f1s = [r["weighted avg"]["f1-score"] for r in all_fold_reports]
        micro_f1s = [r["micro avg"]["f1-score"] for r in all_fold_reports]
        
        print("\n" + "="*50)
        print(f"FINAL 5-FOLD CROSS-VALIDATION SUMMARY")
        print("="*50)
        print(f"Mean Micro F1:    {np.mean(micro_f1s):.4f}")
        print(f"Mean Macro F1:    {np.mean(macro_f1s):.4f}  <-- (Target Metric)")
        print(f"Mean Weighted F1: {np.mean(weighted_f1s):.4f}")
        print("="*50 + "\n")

if __name__ == "__main__":
    TEST_DATA_PATTERN = "data/5_fold_exp/test{}.spacy"
    
    # Run with a lowered threshold (e.g., 0.30) to boost recall
    evaluate_pipeline("models/fold{}_baseline/model-best", TEST_DATA_PATTERN, "RoBERTa-base Baseline", threshold=0.30)
