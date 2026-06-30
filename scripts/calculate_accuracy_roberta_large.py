import spacy
from spacy.tokens import DocBin, Doc
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def get_span_set(doc, key="sc"):
    spans = set()
    if key in doc.spans:
        for span in doc.spans[key]:
            spans.add((span.start, span.end, span.label_))
    return spans

def calculate_strict_accuracy(model_dir_pattern, test_dir_pattern):
    accuracies = []
    
    print(f"\nCalculating Strict Span Accuracy for RoBERTa-large...")
    
    for fold in range(1, 6):
        model_path = model_dir_pattern.format(fold)
        test_data_path = test_dir_pattern.format(fold)

        try:
            nlp = spacy.load(model_path)
            doc_bin = DocBin().from_disk(test_data_path)
            gold_docs = list(doc_bin.get_docs(nlp.vocab))

            clean_docs = []
            for doc in gold_docs:
                clean_docs.append(Doc(nlp.vocab, words=[t.text for t in doc], spaces=[bool(t.whitespace_) for t in doc]))

            pred_docs = list(nlp.pipe(clean_docs))

            tp = 0
            fp = 0
            fn = 0

            for gold_doc, pred_doc in zip(gold_docs, pred_docs):
                gold_spans = get_span_set(gold_doc)
                pred_spans = get_span_set(pred_doc)

                # Calculate TPs and FNs
                for span in gold_spans:
                    if span in pred_spans:
                        tp += 1
                    else:
                        fn += 1
                
                # Calculate FPs
                for span in pred_spans:
                    if span not in gold_spans:
                        fp += 1

            fold_accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
            accuracies.append(fold_accuracy)
            print(f"  Fold {fold} Accuracy: {fold_accuracy:.4f}")

        except Exception as e:
            print(f"  [ERROR] Fold {fold}: {e}")

    final_accuracy = np.mean(accuracies)
    print(f"\nFINAL 5-FOLD ACCURACY: {final_accuracy:.4f}")
    if final_accuracy > 0.7095:
        print("RESULT: RoBERTa-large OUTPERFORMS the RoBERTa+LSTM baseline (0.7095)!")
    else:
        print("RESULT: RoBERTa-large DOES NOT beat the RoBERTa+LSTM baseline (0.7095). Use RoBERTa+LSTM for pseudo-labeling.")

if __name__ == "__main__":
    # Update these paths to match your directory structure
    calculate_strict_accuracy(
        "models/fold{}_roberta_large/model-best", 
        "data/5_fold_exp/test{}.spacy"
    )
