import spacy
from spacy.tokens import DocBin
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Define target labels logically (Engagement classes first, then 'O')
TARGET_LABELS = [
    'ATTRIBUTION', 'CITATION', 'COUNTER', 'DENY', 
    'ENDOPHORIC', 'ENTERTAIN', 'JUSTIFYING', 
    'MONOGLOSS', 'PROCLAIM', 'SOURCES', 'O'
]

def generate_token_confusion_matrix(model_path, test_data_path, output_filename, title):
    """
    Generates a token-level confusion matrix heatmap for sequence taggers.
    Prioritizes Engagement labels over 'O' tags during token flattening to 
    handle overlaps clearly for visualization purposes.
    """
    print(f"Generating Confusion Matrix for: {title}...")
    
    try:
        nlp = spacy.load(model_path)
        doc_bin = DocBin().from_disk(test_data_path)
        gold_docs = list(doc_bin.get_docs(nlp.vocab))
        
        y_true = []
        y_pred = []
        
        # Batch inference for speed
        pred_docs = list(nlp.pipe(gold_docs))
        
        for gold_doc, pred_doc in zip(gold_docs, pred_docs):
            # Flatten Gold
            g_labels = ["O"] * len(gold_doc)
            if "sc" in gold_doc.spans:
                for span in gold_doc.spans["sc"]:
                    for i in range(span.start, span.end):
                        g_labels[i] = span.label_
            y_true.extend(g_labels)
            
            # Flatten Pred
            p_labels = ["O"] * len(pred_doc)
            if "sc" in pred_doc.spans:
                for span in pred_doc.spans["sc"]:
                    for i in range(span.start, span.end):
                        p_labels[i] = span.label_
            y_pred.extend(p_labels)

        # Compute standard confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=TARGET_LABELS)
        
        # Log-normalize the color scale (because 'O' is massively larger than other classes)
        # We add 1 to avoid log(0)
        cm_log = np.log1p(cm)
        
        # Plotting Setup
        plt.figure(figsize=(12, 10))
        sns.set_theme(style="white")
        
        # Draw heatmap
        ax = sns.heatmap(
            cm_log, 
            annot=cm, # Show the actual counts in the boxes
            fmt='d',  # Format as integer
            cmap="Blues", 
            xticklabels=TARGET_LABELS, 
            yticklabels=TARGET_LABELS,
            cbar_kws={'label': 'Log-Scaled Frequency (Colors)'}
        )
        
        # Formatting
        plt.title(f"Token-Level Confusion Matrix: {title}", fontsize=16, pad=20)
        plt.ylabel('True Gold Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save output
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Successfully saved to {output_filename}\n")
        plt.close()
        
    except Exception as e:
        print(f"[ERROR] Failed to generate matrix: {e}")

if __name__ == "__main__":
    # Point this to ONE of your test folds and ONE of your trained models
    # Example using Fold 1:
    TEST_FILE = "data/5_fold_exp/test1.spacy"
    
    # Generate Baseline Matrix
    generate_token_confusion_matrix(
        model_path="models/fold1_baseline/model-best", 
        test_data_path=TEST_FILE, 
        output_filename="confusion_matrix_baseline.png",
        title="RoBERTa-Base (Fold 1)"
    )
    
    # Generate DA-RoBERTa Matrix
    generate_token_confusion_matrix(
        model_path="models/fold1_few_shot_v3/model-best", 
        test_data_path=TEST_FILE, 
        output_filename="confusion_matrix_da_roberta.png",
        title="DA-RoBERTa Few-Shot (Fold 1)"
    )
