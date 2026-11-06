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
from sklearn.feature_selection import SelectKBest, SelectFromModel, RFECV, chi2, mutual_info_classif, VarianceThreshold, f_classif
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

from model_io import save_model

X_train = pd.read_pickle("/lifesciences/pra-0104/jj2157/REALRAW_X_train.pkl")
X_test = pd.read_pickle("/lifesciences/pra-0104/jj2157/REALRAW_X_test.pkl")
train_stx = pd.read_csv("/lifesciences/pra-0104/jj2157/REALRAW_train_stx.csv")['Stx']
test_stx = pd.read_csv("/lifesciences/pra-0104/jj2157/REALRAW_test_stx.csv")['Stx']

skf3 = StratifiedKFold(n_splits=3,
    shuffle=True,
    random_state=123)

# trial 73: prep = none, fs = select_from_model (500 features, Random Forest base estimator)
lp = Pipeline([
    ('fs', SelectFromModel(
        RandomForestClassifier(n_estimators=100,
            class_weight='balanced',
            random_state=123,
            n_jobs=1),
        max_features=500,
        threshold=-np.inf)),
    ('clf', ExtraTreesClassifier(
        n_estimators=223,
        criterion='entropy',
        max_depth=10,
        max_features=0.1,
        min_samples_leaf=2,
        min_samples_split=11,
        bootstrap=False,
        class_weight='balanced',
        random_state=123,
        n_jobs=1,)),])

cv_scores = cross_val_score(
    lp,
    X_train,
    train_stx,
    cv=skf3,
    scoring='balanced_accuracy',
    n_jobs=-1
)

lp.set_params(clf__n_jobs=-1, fs__estimator__n_jobs=-1)   # all cores for the final fit (results unchanged)
lp.fit(X_train, train_stx)
# save_model(lp, "models/tuned_stx_extra_trees.pkl", description="Tuned Extra Trees, Stx prediction")
preds = lp.predict(X_test)
print(f1_score(test_stx, preds, average='macro'))
print(balanced_accuracy_score(test_stx, preds))
print(accuracy_score(test_stx, preds))

labels = sorted(train_stx.unique())
print(confusion_matrix(test_stx, preds, labels=labels))

print(classification_report(
    test_stx,
    preds,
    zero_division=0))

plt.figure(figsize=(7, 7))

sns.heatmap(
    confusion_matrix(test_stx, preds, labels=labels),
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    cbar=True)

plt.xlabel("Predicted Stx")
plt.ylabel("True Stx")
plt.title("Tuned Extra Trees")

plt.tight_layout()
plt.savefig('conf_mat_stx.png', dpi=300, bbox_inches="tight")
plt.show()