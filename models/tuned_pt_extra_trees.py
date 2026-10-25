import os
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer, binarize, LabelEncoder, Binarizer
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import hamming_loss, f1_score, multilabel_confusion_matrix, confusion_matrix, balanced_accuracy_score, accuracy_score
from sklearn.feature_selection import SelectKBest, RFECV, chi2, mutual_info_classif, VarianceThreshold, f_classif
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.preprocessing import binarize, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from boruta import BorutaPy
from skmultilearn.problem_transform import LabelPowerset
from sklearn.cross_decomposition import CCA
from sklearn.inspection import DecisionBoundaryDisplay
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from functools import partial

from model_io import save_model

X_train = pd.read_pickle("/lifesciences/pra-0104/jj2157/finalPTREALRAW_X_train.pkl")
X_test = pd.read_pickle("/lifesciences/pra-0104/jj2157/finalPTREALRAW_X_test.pkl")
train_stx = pd.read_csv("/lifesciences/pra-0104/jj2157/finalPTREALRAW_train_stx.csv")["PT"]
test_stx = pd.read_csv("/lifesciences/pra-0104/jj2157/finalPTREALRAW_test_stx.csv")["PT"]

le = LabelEncoder()

train_stx_labels = le.fit_transform(train_stx)
test_stx_labels = le.transform(test_stx)

skf3 = StratifiedKFold(n_splits=3,
    shuffle=True,
    random_state=123)

lp = Pipeline([
    ('binarize', Binarizer(threshold=0.0)),
    ('vt', VarianceThreshold(threshold=0.001)),
    ('fs', SelectKBest(score_func=partial(mutual_info_classif, random_state=123), k=100000)),
    ('clf', ExtraTreesClassifier(n_estimators=222,
        criterion='gini',
        max_depth=None,
        max_features=0.1,
        min_samples_leaf=4,
        min_samples_split=15,
        bootstrap=False,
        class_weight='balanced_subsample',
        random_state=123,
        n_jobs=1,)),])

cv_scores = cross_val_score(
    lp,
    X_train,
    train_stx,
    cv=skf3,
    scoring='balanced_accuracy',
    n_jobs=-1
).mean()

print(cv_scores)

lp.fit(X_train, train_stx_labels)
save_model(lp, "models/tuned_pt_extra_trees.pkl", label_encoder=le, description="Tuned Extra Trees, PT prediction")
preds = lp.predict(X_test)
print(f1_score(test_stx_labels, preds, average='macro'))
print(balanced_accuracy_score(test_stx_labels, preds))
print(accuracy_score(test_stx_labels, preds))
print(confusion_matrix(test_stx_labels, preds))
pred_labels = le.inverse_transform(preds)


print(classification_report(
    test_stx,
    pred_labels,
    zero_division=0))

labels = le.classes_

plt.figure(figsize=(7, 7))

sns.heatmap(
    confusion_matrix(test_stx_labels,
    preds,
    labels=np.arange(len(labels))),
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    cbar=True)

plt.xlabel("Predicted PT")
plt.ylabel("True PT")
plt.title("Tuned Extra Trees")

plt.tight_layout()
plt.savefig('conf_mat_pt.png', dpi=300, bbox_inches="tight")
plt.show()
