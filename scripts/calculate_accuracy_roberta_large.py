import spacy
from spacy.tokens import DocBin, Doc
import warnings

warnings.filterwarnings("ignore")

def get_span_set(doc, key="sc"):
    spans = set()
    if key in doc.spans:
        for span in doc.spans[key]:
            spans.add((span.start, span.end, span.label_))
    return spans

def evaluate_single_fold(model_path, test_data_path):
    print(f"\nEvaluating Strict Span Accuracy")
    print(f"Model: {model_path}")
    print(f"Test Data: {test_data_path}")
    print("-" * 40)

    try:
        nlp = spacy.load(model_path)
        doc_bin = DocBin().from_disk(test_data_path)
        gold_docs = list(doc_bin.get_docs(nlp.vocab))

        clean_docs = []
        for doc in gold_docs:
            clean_docs.append(Doc(nlp.vocab, words=[t.text for t in doc], spaces=[bool(t.whitespace_) for t in doc]))

        pred_docs = list(nlp.pipe(clean_docs))

        tp, fp, fn = 0, 0, 0

        for gold_doc, pred_doc in zip(gold_docs, pred_docs):
            gold_spans = get_span_set(gold_doc)
            pred_spans = get_span_set(pred_doc)

            for span in gold_spans:
                if span in pred_spans:
                    tp += 1
                else:
                    fn += 1
            
            for span in pred_spans:
                if span not in gold_spans:
                    fp += 1

        accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        
        print(f"True Positives (TP): {tp}")
        print(f"False Positives (FP): {fp}")
        print(f"False Negatives (FN): {fn}")
        print(f"\nSTRICT ACCURACY: {accuracy:.4f}")
        print("=" * 40)

    except Exception as e:
        print(f"  [ERROR]: {e}")

if __name__ == "__main__":
    # 1. Run this in your MAIN environment for RoBERTa-large:
    evaluate_single_fold("models/roberta_large_teacher/model-best", "data/5_fold_exp/test1.spacy")

    # 2. Run this in your EGUCHI_ENV for the baseline:
    # evaluate_single_fold("en_engagement_LSTM", "data/5_fold_exp/test1.spacy")
