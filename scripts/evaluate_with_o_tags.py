import spacy
from spacy.tokens import DocBin, Doc
from sklearn.metrics import classification_report
import numpy as np
import warnings

warnings.filterwarnings("ignore")

ENGAGEMENT_LABELS = [
    'ATTRIBUTION', 'CITATION', 'COUNTER', 'DENY',
    'ENDOPHORIC', 'ENTERTAIN', 'JUSTIFYING',
    'MONOGLOSS', 'PROCLAIM', 'SOURCES'
]
ALL_LABELS = ENGAGEMENT_LABELS + ['O']


def get_span_set(doc, key="sc"):
    spans = set()
    if key in doc.spans:
        for span in doc.spans[key]:
            spans.add((span.start, span.end, span.label_))
    return spans


def evaluate_with_o(model_dir_pattern, test_dir_pattern, model_name):
    macro_f1_scores = []
    weighted_engagement_f1_scores = []

    # Per-label accumulators across folds: label -> list of (P, R, F1) per fold
    per_label_scores = {label: {"precision": [], "recall": [], "f1": []} for label in ALL_LABELS}

    print(f"\n{'='*62}")
    print(f"EVALUATING: {model_name}")
    print(f"METRIC: Hybrid Strict Span + Token-Level O (11-class)")
    print(f"{'='*62}")

    for fold in range(1, 6):
        model_path = model_dir_pattern.format(fold)
        test_data_path = test_dir_pattern.format(fold)

        try:
            nlp = spacy.load(model_path)
            doc_bin = DocBin().from_disk(test_data_path)
            gold_docs = list(doc_bin.get_docs(nlp.vocab))

            clean_docs = []
            for doc in gold_docs:
                words = [t.text for t in doc]
                spaces = [bool(t.whitespace_) for t in doc]
                clean_docs.append(Doc(nlp.vocab, words=words, spaces=spaces))

            print(f"  Running inference for Fold {fold}...")
            pred_docs = list(nlp.pipe(clean_docs))

            tp = {label: 0 for label in ALL_LABELS}
            fp = {label: 0 for label in ALL_LABELS}
            fn = {label: 0 for label in ALL_LABELS}

            for gold_doc, pred_doc in zip(gold_docs, pred_docs):
                doc_len = len(gold_doc)
                gold_spans = get_span_set(gold_doc)
                pred_spans = get_span_set(pred_doc)

                # O: token-level
                gold_covered = set()
                for (s, e, l) in gold_spans:
                    for i in range(s, e):
                        gold_covered.add(i)
                pred_covered = set()
                for (s, e, l) in pred_spans:
                    for i in range(s, e):
                        pred_covered.add(i)
                for i in range(doc_len):
                    g_o = i not in gold_covered
                    p_o = i not in pred_covered
                    if g_o and p_o:
                        tp["O"] += 1
                    elif g_o and not p_o:
                        fp["O"] += 1
                    elif not g_o and p_o:
                        fn["O"] += 1

                # Engagement: strict span-level
                for span in gold_spans:
                    if span in pred_spans:
                        tp[span[2]] += 1
                    else:
                        fn[span[2]] += 1
                for span in pred_spans:
                    if span not in gold_spans:
                        fp[span[2]] += 1

            # Build flat lists
            y_true_flat = []
            y_pred_flat = []
            for label in ALL_LABELS:
                for _ in range(tp[label]):
                    y_true_flat.append(label)
                    y_pred_flat.append(label)
                for _ in range(fp[label]):
                    y_true_flat.append("__none__")
                    y_pred_flat.append(label)
                for _ in range(fn[label]):
                    y_true_flat.append(label)
                    y_pred_flat.append("__none__")

            report_all = classification_report(
                y_true_flat, y_pred_flat,
                labels=ALL_LABELS,
                target_names=ALL_LABELS,
                output_dict=True,
                zero_division=0
            )
            report_all_text = classification_report(
                y_true_flat, y_pred_flat,
                labels=ALL_LABELS,
                target_names=ALL_LABELS,
                zero_division=0
            )
            report_eng = classification_report(
                y_true_flat, y_pred_flat,
                labels=ENGAGEMENT_LABELS,
                target_names=ENGAGEMENT_LABELS,
                output_dict=True,
                zero_division=0
            )

            print(f"\n{'='*22} FOLD {fold} {'='*22}")
            print(report_all_text)
            print(f"  [O-tag counts] TP={tp['O']:,} | FP={fp['O']:,} | FN={fn['O']:,}")

            fold_macro = report_all["macro avg"]["f1-score"]
            fold_weighted_eng = report_eng["weighted avg"]["f1-score"]
            print(f"  Macro F1 (11-class incl. O):       {fold_macro:.4f}")
            print(f"  Weighted F1 (10 engagement only):  {fold_weighted_eng:.4f}")

            macro_f1_scores.append(fold_macro)
            weighted_engagement_f1_scores.append(fold_weighted_eng)

            # Accumulate per-label P, R, F1 for this fold
            for label in ALL_LABELS:
                per_label_scores[label]["precision"].append(report_all[label]["precision"])
                per_label_scores[label]["recall"].append(report_all[label]["recall"])
                per_label_scores[label]["f1"].append(report_all[label]["f1-score"])

        except Exception as e:
            import traceback
            print(f"  [ERROR] Fold {fold}: {e}")
            traceback.print_exc()

    final_macro = np.mean(macro_f1_scores) if macro_f1_scores else 0
    final_weighted_eng = np.mean(weighted_engagement_f1_scores) if weighted_engagement_f1_scores else 0

    print("\n" + "="*62)
    print(f"FINAL 5-FOLD AVERAGE")
    print(f"  Macro F1 (11-class incl. O):      {final_macro:.4f}")
    print(f"  Weighted F1 (engagement only):    {final_weighted_eng:.4f}")
    print("="*62)

    # Per-label averaged summary table
    print(f"\n{'='*62}")
    print(f"PER-LABEL AVERAGE ACROSS ALL 5 FOLDS: {model_name}")
    print(f"{'='*62}")
    print(f"{'Label':<16} {'Avg Precision':>14} {'Avg Recall':>11} {'Avg F1':>8}")
    print(f"{'-'*16} {'-'*14} {'-'*11} {'-'*8}")
    for label in ALL_LABELS:
        avg_p  = np.mean(per_label_scores[label]["precision"])
        avg_r  = np.mean(per_label_scores[label]["recall"])
        avg_f1 = np.mean(per_label_scores[label]["f1"])
        print(f"{label:<16} {avg_p:>14.4f} {avg_r:>11.4f} {avg_f1:>8.4f}")
    print(f"{'-'*16} {'-'*14} {'-'*11} {'-'*8}")
    print(f"{'Macro avg':<16} {np.mean([np.mean(per_label_scores[l]['precision']) for l in ALL_LABELS]):>14.4f} "
          f"{np.mean([np.mean(per_label_scores[l]['recall']) for l in ALL_LABELS]):>11.4f} "
          f"{final_macro:>8.4f}")
    print(f"{'='*62}\n")

    return final_macro, final_weighted_eng


if __name__ == "__main__":
    TEST_DATA_PATTERN = "data/5_fold_exp/test{}.spacy"

    results = {}

    results["RoBERTa-base"] = evaluate_with_o(
        "models/fold{}_baseline/model-best",
        TEST_DATA_PATTERN,
        "RoBERTa-base Baseline"
    )

    results["DA-RoBERTa (zero-shot)"] = evaluate_with_o(
        "models/fold{}_zero_shot/model-best",
        TEST_DATA_PATTERN,
        "Zero-Shot DA-RoBERTa"
    )

    results["DA-RoBERTa (few-shot)"] = evaluate_with_o(
        "models/fold{}_few_shot_v3/model-best",
        TEST_DATA_PATTERN,
        "Few-Shot DA-RoBERTa"
    )

    print("\n" + "="*62)
    print("SUMMARY: HYBRID EVALUATION WITH O-TAG")
    print("="*62)
    print(f"{'Model':<28} {'Macro F1':>12} {'Weighted F1':>14}")
    print(f"{'':28} {'(incl. O)':>12} {'(excl. O)':>14}")
    print("-"*56)
    for model, (macro, weighted) in results.items():
        print(f"{model:<28} {macro:>12.4f} {weighted:>14.4f}")
    print("="*62)
    print("\nEguchi & Kyle (2023) RoBERTa+LSTM:")
    print("  Macro F1 = 0.7208 | Weighted F1 = 0.7283")
    print("  (O-tag included, averaged over engagement categories)")
