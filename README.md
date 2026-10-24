
# Predicting Stx Profiles and Phage Types of Shiga-toxigenic *Escherichia coli* Using k-mer-based Machine Learning

Code accompanying my summer project in Bioinformatics at the University of Bath.

## Project overview

This project investigated the use of k-mer-based machine-learning methods for predicting the Shiga toxin (Stx) profiles and phage types (PTs) of Shiga-toxigenic Escherichia coli (STEC) isolates. Stx prediction involved multilabel classification, while PT prediction involved multiclass classification.

A temporal train-test design was used, with isolates from 2014–2018 used for model development and isolates from 2019 used as an independent test set.

The analysis investigated:
- Baseline machine-learning performance;
- Hyperparameter tuning of various models, including:
  - Random Forest;
  - Logistic Regression;
  - Linear Support Vector Classifier (LinearSVC);
  - Extra Trees;
  - Stochastic Gradient Descent Classifier (SGDClassifier); and
  - Extreme Gradient Boosting (XGBoost).
- Feature selection using multiple techniques, including:
  - Variance thresholding;
  - ANOVA F-test (f_classif);
  - Chi-squared;
  - Mutual Information;
  - Principal Component Analysis (PCA);
  - Recursive Feature Elimination with Cross-Validation (RFECV); and
  - Model-based feature selection (SelectFromModel).
- Approaches for handling class imbalance; and
- Class-level model performance and misclassification patterns.

## Repository structure

```text
.
├── README.md
├── requirements.txt
└── scripts/
    ├── tuned_pt_extra_trees.py
    ├── tuned_stx_linear_svc.py
    ├── baseline_pt_linear_svc.py
    ├── baseline_pt_log_reg.py
    ├── baseline_pt_rf.py
    ├── baseline_stx_linear_svc.py
    ├── baseline_stx_log_reg.py
    ├── baseline_stx_rf.py
    └── data_cleaning.py
```

## Analysis workflow

### 01 — Tuned Extra Trees Classifier (hyperparameter tuning and feature selection) for PT prediction

`tuned_pt_extra_trees.py`

This script implements a machine learning pipeline using a tuned Extra Trees Classifier to predict 'PT' labels from training and test datasets. It includes cross-validation, model fitting, and evaluation metrics such as F1 score and confusion matrix. In addition to hyperparameter tuning, binarisation of k-mer features and feature selection using variance thresholding and mutual information improved predictive performance for this particular task.

### 02 — Tuned LinearSVC (hyperparameter tuning and feature selection) for Stx prediction

`tuned_stx_linear_svc.py`

This script implements a machine learning pipeline using a tuned LinearSVC to predict 'Stx' labels from training and test datasets. It includes cross-validation, model fitting, and evaluation metrics such as F1 score and confusion matrix. In addition to hyperparameter tuning, binarisation of k-mer features and feature selection using variance thresholding and the ANOVA F-test (f_classif) improved predictive performance for this particular task.

### 03 — Baseline LinearSVC (no hyperparameter tuning or feature selection) for PT prediction

`baseline_pt_linear_svc.py`

This script implements a machine learning pipeline using LinearSVC to predict 'PT' labels from training and test datasets. It includes cross-validation, model fitting, and evaluation metrics such as F1 score and confusion matrix.

### 04 — Baseline Logistic Regression (no hyperparameter tuning or feature selection) for PT prediction

`baseline_pt_log_reg.py`

This script implements a machine learning pipeline using Logistic Regression to predict 'PT' labels from training and test datasets. It includes cross-validation, model fitting, and evaluation metrics such as F1 score and confusion matrix.

### 05 — Baseline Random Forest Classifier (no hyperparameter tuning or feature selection) for PT prediction

`baseline_pt_rf.py`

This script implements a machine learning pipeline using a Random Forest Classifier to predict 'PT' labels from training and test datasets. It includes cross-validation, model fitting, and evaluation metrics such as F1 score and confusion matrix.

### 06 — Baseline LinearSVC (no hyperparameter tuning or feature selection) for Stx prediction

`baseline_stx_linear_svc.py`

This script implements a machine learning pipeline using LinearSVC to predict 'Stx' labels from training and test datasets. It includes cross-validation, model fitting, and evaluation metrics such as F1 score and confusion matrix.

### 07 — Baseline Logistic Regression (no hyperparameter tuning or feature selection) for Stx prediction

`baseline_stx_log_reg.py`

This script implements a machine learning pipeline using Logistic Regression to predict 'Stx' labels from training and test datasets. It includes cross-validation, model fitting, and evaluation metrics such as F1 score and confusion matrix.

### 08 — Baseline Random Forest Classifier (no hyperparameter tuning or feature selection) for Stx prediction

`baseline_stx_rf.py`

This script implements a machine learning pipeline using a Random Forest Classifier to predict 'Stx' labels from training and test datasets. It includes cross-validation, model fitting, and evaluation metrics such as F1 score and confusion matrix.

### 09 — Cleaning the dataset

`data_cleaning.py`

This script preprocesses the datasets by removing unnecessary rows and columns, then saves the cleaned data in pickle and CSV formats for analysis.

## Model evaluation

Model performance was assessed using:

- accuracy;
- balanced accuracy;
- macro F1 score;
- classification reports; and
- confusion matrices.

Balanced accuracy and macro F1 were considered alongside overall accuracy because of imbalance between classes.

## Data

The complete k-mer matrices and associated isolate metadata used in this study are not included in this repository due to file size and data-access considerations.

The analysis scripts expect the following input files:

- `train_metadata`
- `test_metadata`
- `assembly_kmer_RF_STEC_14-18.txt`
- `assembly_kmer_RF_STEC_19.txt`

The training dataset contains isolates from 2014–2018, while the independent test dataset contains isolates from 2019.

K-mer matrices are aligned with metadata using isolate accession identifiers before model training and evaluation.

## Software

The analysis was performed in Python using:

- NumPy
- pandas
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn
- boruta
- xgboost

Computationally intensive analyses were run on the University of Bath high-performance computing (HPC) cluster.

## Reproducibility

Random seeds are specified within the analysis scripts where applicable.

## Author

Jeevan Singh Juss

BSc Mathematics

University of Bath
