"""Example: load a saved model and predict on a k-mer matrix.

Usage (from the repository root):
    python scripts/load_and_predict_example.py models/tuned_pt_extra_trees.pkl X_test.pkl
    python scripts/load_and_predict_example.py models/tuned_pt_extra_trees.pkl X_test.pkl --labels test_labels.csv --column PT

X must be a pickled pandas DataFrame: rows = isolates, columns = k-mers.
"""
import argparse

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from model_io import load_model, predict

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("kmer_matrix")
parser.add_argument("--labels", help="optional CSV of true labels (same row order as X)")
parser.add_argument("--column", default="PT", help="label column in the CSV")
args = parser.parse_args()

bundle = load_model(args.model)
print("Trained with:", bundle["versions"])
print("Classes:", bundle["classes"])

X = pd.read_pickle(args.kmer_matrix)
preds = predict(bundle, X)
print(pd.Series(preds, index=X.index).value_counts())

if args.labels:
    y = pd.read_csv(args.labels)[args.column]
    print("accuracy         ", round(accuracy_score(y, preds), 3))
    print("balanced accuracy", round(balanced_accuracy_score(y, preds), 3))
    print("macro F1         ", round(f1_score(y, preds, average="macro"), 3))
