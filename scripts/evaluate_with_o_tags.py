import spacy
from spacy.tokens import DocBin, Doc
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def evaluate_token_level_o_tags(model_dir_pattern, test_dir_pattern, model_name):
    macro_f1_scores = []
    weighted_f1_scores = []
    
    print(f"\n==================================================")
    print(f"EVALUATING: {model_name}")
    print(f"==================================================")
    
    for fold in range(1, 6):
        model_path = model_dir_pattern.format(fold)
        test_data_path = test_dir_pattern.format(fold)
        
        try:
            nlp = spacy.load(model_path)
            doc_bin = DocBin().from_disk(test_data_path)
            gold_docs = list(doc_bin.get_docs(nlp.vocab))
            
            # 1. Recreate EXACT tokenization to bypass the tokenizer and preserve boundaries
            clean_docs = []
            for doc in gold_docs:
                words = [t.text for t in doc]
                spaces = [bool(t.whitespace_) for t in doc]
                clean_docs.append(Doc(nlp.vocab, words=words, spaces=spaces))
            
            print(f"  Running batched inference for Fold {fold}...")
            pred_docs = list(nlp.pipe(clean_docs))
            
            y_true = []
            y_pred = []
            
            for gold_doc, pred_doc in zip(gold_docs, pred_docs):
                doc_len = len(gold_doc)
                g_toks = [set() for _ in range(doc_len)]
                p_toks = [set() for _ in range(doc_len)]
                
                # 2. Map Gold Spans
                if "sc" in gold_doc.spans:
                    for span in gold_doc.spans["sc"]:
                        for i in range(span.start, span.end):
                            g_toks[i].add(span.label_)
                            
                # 3. Map Predicted Spans (Indices match perfectly because we bypassed the tokenizer)
                if "sc" in pred_doc.spans:
                    for span in pred_doc.spans["sc"]:
                        for i in range(span.start, span.end):
                            p_toks[i].add(span.label_)
                            
                # 4. Inject 'O' Tags for background tokens
                for i in range(doc_len):
                    if not g_toks[i]: g_toks[i].add("O")
                    if not p_toks[i]: p_toks[i].add("O")
                    
                y_true.extend([list(s) for s in g_toks])
                y_pred.extend([list(s) for s in p_toks])
                
            # 5. Multi-Label Math (Solves Overlap Collisions)
            mlb = MultiLabelBinarizer()
            y_true_bin = mlb.fit_transform(y_true)
            y_pred_bin = mlb.transform(y_pred)
            
            report = classification_report(y_true_bin, y_pred_bin, target_names=mlb.classes_, output_dict=True, zero_division=0)
            
            fold_macro = report["macro avg"]["f1-score"]
            fold_weighted = report["weighted avg"]["f1-score"]
            
            macro_f1_scores.append(fold_macro)
            weighted_f1_scores.append(fold_weighted)
            
            print(f"  -> Fold {fold} Macro F1: {fold_macro:.4f} | Weighted F1: {fold_weighted:.4f}")
            
        except Exception as e:
            print(f"  [ERROR] Could not process Fold {fold}: {e}")

    final_macro = np.mean(macro_f1_scores) if macro_f1_scores else 0
    final_weighted = np.mean(weighted_f1_scores) if weighted_f1_scores else 0
    
    print("-----------------------------------------------------------")
    print(f"FINAL 5-FOLD AVERAGE -> Macro F1: {final_macro:.4f} | Weighted F1: {final_weighted:.4f}")
    print("-----------------------------------------------------------\n")

if __name__ == "__main__":
    TEST_DATA_PATTERN = "data/5_fold_exp/test{}.spacy"
    
    evaluate_token_level_o_tags("models/fold{}_baseline/model-best", TEST_DATA_PATTERN, "RoBERTa-base Baseline")
    evaluate_token_level_o_tags("models/fold{}_zero_shot/model-best", TEST_DATA_PATTERN, "Zero-Shot DA-RoBERTa")
    evaluate_token_level_o_tags("models/fold{}_few_shot_v3/model-best", TEST_DATA_PATTERN, "Few-Shot DA-RoBERTa")
