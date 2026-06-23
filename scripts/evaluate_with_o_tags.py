import spacy
from spacy.tokens import DocBin, Doc
from sklearn.metrics import classification_report
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# All 11 classes: 10 engagement labels + O
TARGET_LABELS = [
    'ATTRIBUTION', 'CITATION', 'COUNTER', 'DENY',
    'ENDOPHORIC', 'ENTERTAIN', 'JUSTIFYING',
    'MONOGLOSS', 'O', 'PROCLAIM', 'SOURCES'
]

def get_span_set(doc, key="sc"):
    """
    Extract spans as a set of (start, end, label) tuples.
    This is the same unit spacy evaluate uses for strict span F1:
    a span is correct only if start token, end token, AND label all match exactly.
    """
    spans = set()
    if key in doc.spans:
        for span in doc.spans[key]:
            spans.add((span.start, span.end, span.label_))
    return spans


def evaluate_strict_span_with_o(model_dir_pattern, test_dir_pattern, model_name):
    macro_f1_scores = []
    weighted_f1_scores = []

    print(f"\n{'='*58}")
    print(f"EVALUATING: {model_name}")
    print(f"METRIC: Strict Span-Level F1 (11-class including O)")
    print(f"{'='*58}")

    for fold in range(1, 6):
        model_path = model_dir_pattern.format(fold)
        test_data_path = test_dir_pattern.format(fold)

        try:
            nlp = spacy.load(model_path)
            doc_bin = DocBin().from_disk(test_data_path)
            gold_docs = list(doc_bin.get_docs(nlp.vocab))

            # Strip gold annotations from inference input
            clean_docs = []
            for doc in gold_docs:
                words = [t.text for t in doc]
                spaces = [bool(t.whitespace_) for t in doc]
                clean_docs.append(Doc(nlp.vocab, words=words, spaces=spaces))

            print(f"  Running inference for Fold {fold}...")
            pred_docs = list(nlp.pipe(clean_docs))

            # Per-class TP, FP, FN counters
            tp = {label: 0 for label in TARGET_LABELS}
            fp = {label: 0 for label in TARGET_LABELS}
            fn = {label: 0 for label in TARGET_LABELS}

            for gold_doc, pred_doc in zip(gold_docs, pred_docs):
                gold_spans = get_span_set(gold_doc)
                pred_spans = get_span_set(pred_doc)

                # --- O-tag logic ---
                # A sentence is "O" if it contains NO engagement spans at all.
                # We treat the sentence itself as the O unit (one O per sentence).
                # This mirrors how Eguchi & Kyle's evaluation inflates via O:
                # sentences with no spans are correctly classified as background.
                gold_has_no_spans = len(gold_spans) == 0
                pred_has_no_spans = len(pred_spans) == 0

                if gold_has_no_spans and pred_has_no_spans:
                    tp["O"] += 1          # Both agree: no engagement here
                elif gold_has_no_spans and not pred_has_no_spans:
                    fp["O"] += 1          # Model predicted spans where there are none (false alarm)
                    # Each wrongly predicted label is already counted below as FP
                elif not gold_has_no_spans and pred_has_no_spans:
                    fn["O"] += 1          # Model missed all spans — entire sentence treated as O

                # --- Strict span matching (exact start, end, label) ---
                for span in gold_spans:
                    if span in pred_spans:
                        tp[span[2]] += 1
                    else:
                        fn[span[2]] += 1

                for span in pred_spans:
                    if span not in gold_spans:
                        fp[span[2]] += 1

            # Compute per-class F1
            y_true_flat = []
            y_pred_flat = []

            # Build flat label lists from TP/FP/FN for classification_report
            # TP: label appears in both true and pred
            # FP: label appears in pred only
            # FN: label appears in true only
            for label in TARGET_LABELS:
                # TP instances
                for _ in range(tp[label]):
                    y_true_flat.append(label)
                    y_pred_flat.append(label)
                # FP instances
                for _ in range(fp[label]):
                    y_true_flat.append("__none__")
                    y_pred_flat.append(label)
                # FN instances
                for _ in range(fn[label]):
                    y_true_flat.append(label)
                    y_pred_flat.append("__none__")

            all_labels = TARGET_LABELS + ["__none__"]

            report_dict = classification_report(
                y_true_flat, y_pred_flat,
                labels=TARGET_LABELS,
                target_names=TARGET_LABELS,
                output_dict=True,
                zero_division=0
            )
            report_text = classification_report(
                y_true_flat, y_pred_flat,
                labels=TARGET_LABELS,
                target_names=TARGET_LABELS,
                zero_division=0
            )

            print(f"\n{'='*20} FOLD {fold} {'='*20}")
            print(report_text)

            # Print raw TP/FP/FN for O to aid verification
            print(f"  [O-tag counts] TP={tp['O']} | FP={fp['O']} | FN={fn['O']}")

            fold_macro = report_dict["macro avg"]["f1-score"]
            fold_weighted = report_dict["weighted avg"]["f1-score"]

            macro_f1_scores.append(fold_macro)
            weighted_f1_scores.append(fold_weighted)

        except Exception as e:
            import traceback
            print(f"  [ERROR] Could not process Fold {fold}: {e}")
            traceback.print_exc()

    final_macro = np.mean(macro_f1_scores) if macro_f1_scores else 0
    final_weighted = np.mean(weighted_f1_scores) if weighted_f1_scores else 0

    print("\n" + "="*58)
    print(f"FINAL 5-FOLD AVERAGE")
    print(f"  Macro F1    : {final_macro:.4f}")
    print(f"  Weighted F1 : {final_weighted:.4f}")
    print("="*58 + "\n")

    return final_macro, final_weighted


if __name__ == "__main__":
    TEST_DATA_PATTERN = "data/5_fold_exp/test{}.spacy"

    results = {}

    results["RoBERTa-base"] = evaluate_strict_span_with_o(
        "models/fold{}_baseline/model-best",
        TEST_DATA_PATTERN,
        "RoBERTa-base Baseline"
    )

    results["DA-RoBERTa (zero-shot)"] = evaluate_strict_span_with_o(
        "models/fold{}_zero_shot/model-best",
        TEST_DATA_PATTERN,
        "Zero-Shot DA-RoBERTa"
    )

    results["DA-RoBERTa (few-shot)"] = evaluate_strict_span_with_o(
        "models/fold{}_few_shot_v3/model-best",
        TEST_DATA_PATTERN,
        "Few-Shot DA-RoBERTa"
    )

    print("\n" + "="*58)
    print("SUMMARY: STRICT SPAN F1 WITH O-TAG (11-CLASS)")
    print("="*58)
    print(f"{'Model':<28} {'Macro F1':>10} {'Weighted F1':>12}")
    print("-"*52)
    for model, (macro, weighted) in results.items():
        print(f"{model:<28} {macro:>10.4f} {weighted:>12.4f}")
    print("="*58)
