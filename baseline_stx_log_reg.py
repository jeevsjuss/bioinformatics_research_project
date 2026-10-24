import os
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer, binarize, LabelEncoder
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import hamming_loss, f1_score, multilabel_confusion_matrix, confusion_matrix, balanced_accuracy_score, accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, RFECV, chi2, mutual_info_classif, VarianceThreshold
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

X_train = pd.read_pickle("/lifesciences/pra-0104/jj2157/REALRAW_X_train.pkl")
X_test = pd.read_pickle("/lifesciences/pra-0104/jj2157/REALRAW_X_test.pkl")
train_stx = pd.read_csv("/lifesciences/pra-0104/jj2157/REALRAW_train_stx.csv")['Stx']
test_stx = pd.read_csv("/lifesciences/pra-0104/jj2157/REALRAW_test_stx.csv")['Stx']

skf3 = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=123)

lp = Pipeline([('lr', LogisticRegression(random_state=123))])

cv_scores = cross_val_score(
    lp,
    X_train,
    train_stx,
    cv=skf3,
    scoring='balanced_accuracy',
    n_jobs=-1).mean()

print(cv_scores)

lp.fit(X_train, train_stx)
preds = lp.predict(X_test)
print(f1_score(test_stx, preds, average='macro'))
print(balanced_accuracy_score(test_stx, preds))
print(accuracy_score(test_stx, preds))
print(confusion_matrix(test_stx, preds))

print(classification_report(
    test_stx,
    preds,
    zero_division=0))
