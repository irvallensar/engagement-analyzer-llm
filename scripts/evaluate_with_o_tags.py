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

ENGAGEMENT_LABELS = set(TARGET_LABELS) - {"O"}


def get_token_label_map(doc, key="sc"):
    """
    Build a token-index -> set-of-labels mapping.
    Replicates IOB token-level evaluation exactly:
    every token inside a span is one instance of that label.
    Overlapping spans give a token multiple labels (multi-label).
    Tokens not covered by any span become O.
    """
    token_labels = {}
    if key in doc.spans:
        for span in doc.spans[key]:
            for i in range(span.start, span.end):
                if i not in token_labels:
                    token_labels[i] = set()
                token_labels[i].add(span.label_)
    return token_labels


def evaluate_iob_equivalent(model_dir_pattern, test_dir_pattern, model_name):
    macro_f1_scores = []
    weighted_f1_scores = []

    print(f"\n{'='*62}")
    print(f"EVALUATING: {model_name}")
    print(f"METRIC: Full Token-Level IOB-Equivalent (11-class including O)")
    print(f"{'='*62}")

    for fold in range(1, 6):
        model_path = model_dir_pattern.format(fold)
        test_data_path = test_dir_pattern.format(fold)

        try:
            nlp = spacy.load(model_path)
            doc_bin = DocBin().from_disk(test_data_path)
            gold_docs = list(doc_bin.get_docs(nlp.vocab))

            # Strip gold annotations so model cannot cheat
            clean_docs = []
            for doc in gold_docs:
                words = [t.text for t in doc]
                spaces = [bool(t.whitespace_) for t in doc]
                clean_docs.append(Doc(nlp.vocab, words=words, spaces=spaces))

            print(f"  Running inference for Fold {fold}...")
            pred_docs = list(nlp.pipe(clean_docs))

            # Per-class TP, FP, FN counters — all token-level
            tp = {label: 0 for label in TARGET_LABELS}
            fp = {label: 0 for label in TARGET_LABELS}
            fn = {label: 0 for label in TARGET_LABELS}

            for gold_doc, pred_doc in zip(gold_docs, pred_docs):
                doc_len = len(gold_doc)

                # Build token-level label maps for gold and pred
                gold_map = get_token_label_map(gold_doc)
                pred_map = get_token_label_map(pred_doc)

                for i in range(doc_len):
                    gold_labels = gold_map.get(i, set())
                    pred_labels = pred_map.get(i, set())

                    # O token: neither gold nor pred has any engagement label
                    gold_is_o = len(gold_labels) == 0
                    pred_is_o = len(pred_labels) == 0

                    # --- O-tag counting ---
                    if gold_is_o and pred_is_o:
                        tp["O"] += 1
                    elif gold_is_o and not pred_is_o:
                        fp["O"] += 1
                    elif not gold_is_o and pred_is_o:
                        fn["O"] += 1
                    # if both have engagement labels, O is neither TP/FP/FN

                    # --- Engagement label counting (token-level, multi-label) ---
                    for label in ENGAGEMENT_LABELS:
                        in_gold = label in gold_labels
                        in_pred = label in pred_labels

                        if in_gold and in_pred:
                            tp[label] += 1
                        elif in_gold and not in_pred:
                            fn[label] += 1
                        elif not in_gold and in_pred:
                            fp[label] += 1
                        # true negative (neither gold nor pred): not counted

            # Build flat lists for classification_report from TP/FP/FN
            y_true_flat = []
            y_pred_flat = []

            for label in TARGET_LABELS:
                for _ in range(tp[label]):
                    y_true_flat.append(label)
                    y_pred_flat.append(label)
                for _ in range(fp[label]):
                    y_true_flat.append("__none__")
                    y_pred_flat.append(label)
                for _ in range(fn[label]):
                    y_true_flat.append(label)
                    y_pred_flat.append("__none__")

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

            print(f"\n{'='*22} FOLD {fold} {'='*22}")
            print(report_text)
            print(f"  [O-tag counts] TP={tp['O']:,} | FP={fp['O']:,} | FN={fn['O']:,}")

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

    print("\n" + "="*62)
    print(f"FINAL 5-FOLD AVERAGE")
    print(f"  Macro F1    : {final_macro:.4f}")
    print(f"  Weighted F1 : {final_weighted:.4f}")
    print("="*62 + "\n")

    return final_macro, final_weighted


if __name__ == "__main__":
    TEST_DATA_PATTERN = "data/5_fold_exp/test{}.spacy"

    results = {}

    results["RoBERTa-base"] = evaluate_iob_equivalent(
        "models/fold{}_baseline/model-best",
        TEST_DATA_PATTERN,
        "RoBERTa-base Baseline"
    )

    results["DA-RoBERTa (zero-shot)"] = evaluate_iob_equivalent(
        "models/fold{}_zero_shot/model-best",
        TEST_DATA_PATTERN,
        "Zero-Shot DA-RoBERTa"
    )

    results["DA-RoBERTa (few-shot)"] = evaluate_iob_equivalent(
        "models/fold{}_few_shot_v3/model-best",
        TEST_DATA_PATTERN,
        "Few-Shot DA-RoBERTa"
    )

    print("\n" + "="*62)
    print("SUMMARY: IOB-EQUIVALENT TOKEN-LEVEL F1 (11-CLASS WITH O)")
    print("="*62)
    print(f"{'Model':<28} {'Macro F1':>10} {'Weighted F1':>12}")
    print("-"*52)
    for model, (macro, weighted) in results.items():
        print(f"{model:<28} {macro:>10.4f} {weighted:>12.4f}")
    print("="*62)
    print("\nNote: Eguchi & Kyle (2023) Macro F1=0.7208, Weighted F1=0.7283")
    print("      (RoBERTa+LSTM, O-tag included, token-level IOB evaluation)")
