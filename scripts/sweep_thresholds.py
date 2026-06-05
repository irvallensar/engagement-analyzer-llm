import spacy
from spacy.tokens import DocBin
from spacy.training.example import Example

def main():
    # We will test this on your Fold 1 Few-Shot V3 model. 
    # (You can change this path to your Zero-Shot model as well to compare!)
    model_path = "./models/fold1_few_shot_v3/model-best"
    test_path = "./data/5_fold_exp/test1.spacy"

    print(f"Loading model from: {model_path}")
    nlp = spacy.load(model_path)
    
    print(f"Loading test data from: {test_path}")
    doc_bin = DocBin().from_disk(test_path)
    ref_docs = list(doc_bin.get_docs(nlp.vocab))

    # We will test cutoffs from very forgiving (0.1) to very strict (0.8)
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    print("\n" + "="*60)
    print(f"  THRESHOLD SWEEP FOR FOLD 1")
    print("="*60)
    print(f"{'Threshold':<12} | {'Macro F1':<12} | {'Precision':<12} | {'Recall':<12}")
    print("-" * 60)

    for t in thresholds:
        # Dynamically change the neural network's confidence cutoff
        nlp.get_pipe("spancat").cfg["threshold"] = t

        examples = []
        for doc in ref_docs:
            # Pass the raw text back through the adjusted pipeline
            predicted = nlp.make_doc(doc.text)
            predicted = nlp(predicted)
            examples.append(Example(predicted, doc))

        # Evaluate the adjusted predictions against the gold standards
        scores = nlp.evaluate(examples)
        
        # Extract the specific spancat metrics
        f1 = scores.get("spans_sc_f", 0.0) * 100
        p = scores.get("spans_sc_p", 0.0) * 100
        r = scores.get("spans_sc_r", 0.0) * 100

        print(f"{t:<12.1f} | {f1:<12.2f} | {p:<12.2f} | {r:<12.2f}")
        
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
