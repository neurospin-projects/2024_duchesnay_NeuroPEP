import os
import os.path
import sys
# import tempfile
import urllib.request
import glob
import time
from pathlib import Path
from shutil import copyfile, make_archive, unpack_archive, move

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Models
from sklearn.decomposition import PCA
import sklearn.linear_model as lm
import sklearn.svm as svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Univariate statistics
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms

# Metrics
import sklearn.metrics as metrics

# Resampling
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from sklearn import preprocessing
from sklearn.pipeline import make_pipeline


from sklearn.base import clone
#from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier

# %% config

WD = "/home/ed203246/data/2024_NeuroPEP"
sourcedata_filename = "sourcedata/NeuroPEP2024_database_19082024.xlsx"
data_filename = "derivatives/NeuroPEP2024_database_19082024.csv"

os.chdir(WD)

# T0
# Gradient neurodéveloppemental (score quantitatif /32) composé de : 
cols_t0_gradNeuroDev = [
'Epilepsie',
'atcd_TND1',
'Dyslexie',# (je l'ai rajouté, c'est un oubli de ma part)
'DTD1_RS',
'DTD2_COM',
'DTD3_CPT',
'DTD4_DVP',
'Encopresie']

# Gradient DSM (score quantitatif de 0 à 6) composé de tous les troubles neurodév du DSM5 : 
cols_t0_DSM = [
'DSM_MOT',
'DSM_COM',
'DSM_DI',
'DSM_TSA',
'DSM_TDAH',
'DSM_DYS']

# Et +/- l'étude des troubles neurodév sur T1 et T2 de manière indépendante, avec notamment
#cols_t0_troubleNeuroDev= [
#    'DSM_TSA',
#    'DSM_TDAH']

cols_t0 = ['gradNeuroDev', 'DSM_Sum'] + cols_t0_DSM


# T1

cols_t1 = [
'Catatonie', #  (5 patients sur les 49 ont eu ce symptôme)
'CAARMS1s', #  : gradient de 0 à 6 coté pour tous les patients (n = 49)
'CAARMS2s', #  : idem
'CAARMS3s', #  : idem
'CAARMS4s', #  : idem
'CDI' #  : gradient de 0 à 54, on a des données manquantes, le n est à 38
]

# T1

cols_t2 = [
'psycho_PR', # : patients présentants des symptômes psychotiques dits "de premier rang" (plus spécifiques de schizophrénie, ceux pour lesquels on aimerait qu'il n'y ai pas d'association avec nos patients les plus neurodév) (9 patients au total)
'psycho_autres', # : tout autre symptôme psychotique (hallucinations notamment, délire de persécution, etc), représente 23 patients de la cohorte
'thymique', # : symptôme affectant l'humeur, concerne 17 patients
'remission', # : concerne 6 patients
'trauma_cpx' # : concerne 7 patients
]
# Seule la catégorie remission pourrait être exclusive des autres.
# Si cela pose problème, qu'en dis-tu Julie, on peut la virer et remplacer éventuellement par les scores CGI (de 0 à 7) et PSP (de 0 à 100) ?


# %% Utils

def cross_validate_classif_summary(cv_res):
    N_FOLDS = len(cv_res['test_accuracy'])
    acc = cv_res['test_accuracy'].mean()
    acc_se = cv_res['test_accuracy'].std() / np.sqrt(N_FOLDS)

    bacc = cv_res['test_balanced_accuracy'].mean()
    bacc_se = cv_res['test_balanced_accuracy'].std() / np.sqrt(N_FOLDS)

    auc = cv_res['test_roc_auc'].mean()
    auc_se = cv_res['test_roc_auc'].std() / np.sqrt(N_FOLDS)
    return acc, acc_se, bacc, bacc_se, auc, auc_se


def pipeline_split_body_head(pipe):
    return Pipeline(pipe.steps[:-1]) , pipe.steps[-1][1]


def linear_model_split(lm, where):
    from copy import deepcopy
    lm1, lm2 = deepcopy(lm), deepcopy(lm)
    lm1.coef_[0, where:] = 0
    lm2.coef_[0, :where] = 0
    return lm1, lm2


# %% Load data

data = pd.read_csv(os.path.join(WD, "data", data_filename))

# Hypothèses deux trajectoires
# NeuroDev++  => CAARMS1.2++ & Catatonique++ => psycho_PR=0
# NeuroDev-- => CAARMS1.1 & CAARMS1.4 => psycho_PR=1
# Predicteur(T0 + T1) => psycho_PR


data = data[data["psycho_PR"].notnull()]
[[x, data[x].isnull().sum()] for x in data.columns]

# Imput missing
# ['CDI', 9], 
data.loc[data['CDI'].isnull(), 'CDI'] = data.loc[data['CDI'].notnull(), 'CDI'].mean()

X0 = data[cols_t0]
X1 = data[cols_t1]
y = data["psycho_PR"]


# %% Dictionary of models with model selection grid-search CV
# ===========================================================

N_FOLDS = 5
N_FOLDS_VAL = 3
cv_train = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
cv_val = StratifiedKFold(n_splits=N_FOLDS_VAL, shuffle=True, random_state=0)

mlp_param_grid = {"hidden_layer_sizes":
                  [(100, ), (50, ), (25, ), (10, ), (5, ),          # 1 hidden layer
                   (100, 50, ), (50, 25, ), (25, 10, ), (10, 5, ),  # 2 hidden layers
                   (100, 50, 25, ), (50, 25, 10, ), (25, 10, 5, )], # 3 hidden layers
                  "activation": ["relu"], "solver": ["sgd"], 'alpha': [0.0001]}

models = dict(
    lrl2_cv=make_pipeline(
        preprocessing.StandardScaler(),
        # preprocessing.MinMaxScaler(),
        GridSearchCV(lm.LogisticRegression(class_weight='balanced'),
                     {'C': 10. ** np.arange(-3, 1)},
                     cv=cv_val, n_jobs=N_FOLDS_VAL)),

    lrenet_cv=make_pipeline(
        preprocessing.StandardScaler(),
        # preprocessing.MinMaxScaler(),
        GridSearchCV(estimator=lm.SGDClassifier(loss='log_loss', penalty='elasticnet', class_weight='balanced'),
                 param_grid={'alpha': 10. ** np.arange(-1, 3),
                             'l1_ratio': [.1, .5, .9]},
                             cv=cv_val, n_jobs=N_FOLDS_VAL)),

    svmrbf_cv=make_pipeline(
        # preprocessing.StandardScaler(),
        preprocessing.MinMaxScaler(),
        GridSearchCV(svm.SVC(class_weight='balanced'),
                     # {'kernel': ['poly', 'rbf'], 'C': 10. ** np.arange(-3, 3)},
                     {'kernel': ['rbf'], 'C': 10. ** np.arange(-1, 2)},
                     cv=cv_val, n_jobs=N_FOLDS_VAL)),

    forest_cv=make_pipeline(
        # preprocessing.StandardScaler(),
        preprocessing.MinMaxScaler(),
        GridSearchCV(RandomForestClassifier(random_state=1, class_weight='balanced'),
                     {"n_estimators": [10, 100]},
                     cv=cv_val, n_jobs=N_FOLDS_VAL)),

    gb_cv=make_pipeline(
        preprocessing.MinMaxScaler(),
        GridSearchCV(estimator=GradientBoostingClassifier(random_state=1),
                     param_grid={"n_estimators": [10, 100]},
                     cv=cv_val, n_jobs=N_FOLDS_VAL)),

    mlp_cv=make_pipeline(
        # preprocessing.StandardScaler(),
        preprocessing.MinMaxScaler(),
        GridSearchCV(estimator=MLPClassifier(random_state=1),
                     param_grid=mlp_param_grid,
                     cv=cv_val, n_jobs=N_FOLDS_VAL)))


#models.pop('mlp_cv', None)
print(models.keys())

# %% Run

res = list()

for model_str in models:
    print("#### %s" % model_str)
    cv_res = cross_validate(estimator=models[model_str], X=X0, y=y, cv=cv_train,
                            n_jobs=N_FOLDS,
                            scoring=['accuracy', 'balanced_accuracy',
                                    'roc_auc'])

    print("X0, CV ACC:%.3f±%.3f, bACC:%.3f±%.3f, AUC:%.3f±%.3f" % cross_validate_classif_summary(cv_res))
    # X0, CV ACC:0.561±0.084, bACC:0.652±0.081, AUC:0.535±0.092
    res.append(["X0", model_str] + list(cross_validate_classif_summary(cv_res)))


    cv_res = cross_validate(estimator=models[model_str], X=X1, y=y, cv=cv_train,
                            n_jobs=N_FOLDS,
                            scoring=['accuracy', 'balanced_accuracy',
                                    'roc_auc'])

    print("X1: CV ACC:%.3f±%.3f, bACC:%.3f±%.3f, AUC:%.3f±%.3f" % cross_validate_classif_summary(cv_res))
    # X1: CV ACC:0.742±0.041, bACC:0.733±0.078, AUC:0.781±0.085
    res.append(["X1", model_str] + list(cross_validate_classif_summary(cv_res)))



"""   
   Data      Model  ACC-mean  ACC-sd  bACC-mean  bACC-sd  AUC-mean  AUC-sd
0    X0    lrl2_cv     0.561   0.084      0.652    0.081     0.535   0.092
1    X1    lrl2_cv     0.742   0.041      0.733    0.078     0.781   0.085
2    X0  lrenet_cv     0.569   0.128      0.500    0.000     0.500   0.000
3    X1  lrenet_cv     0.792   0.019      0.500    0.000     0.500   0.000
4    X0  svmrbf_cv     0.464   0.110      0.560    0.074     0.723   0.063
5    X1  svmrbf_cv     0.700   0.051      0.562    0.084     0.762   0.067
6    X0  forest_cv     0.700   0.036      0.633    0.056     0.618   0.118
7    X1  forest_cv     0.744   0.051      0.505    0.056     0.660   0.079
8    X0      gb_cv     0.725   0.066      0.457    0.038     0.542   0.062
9    X1      gb_cv     0.742   0.041      0.626    0.062     0.729   0.075
10   X0     mlp_cv     0.792   0.019      0.500    0.000     0.735   0.088
11   X1     mlp_cv     0.792   0.019      0.500    0.000     0.252   0.086
"""


# On garde le linéaire.
# %% Keep LRL2 : Stacking/concatenation of LRL2
# ============================================

X = np.concatenate([X0, X1], axis=1)

# Pipelines with columns selector
ss_ = preprocessing.StandardScaler()

# Test to build column selectors
cs_t0 = Pipeline([("selector", ColumnTransformer([("selector", "passthrough", cols_t0)], remainder="drop")),
                  ("scaler", preprocessing.StandardScaler())
                  ])
assert np.all(cs_t0.fit_transform(data) == ss_.fit_transform(data[cols_t0]))

cs_t1 = Pipeline([("selector", ColumnTransformer([("selector", "passthrough", cols_t1)], remainder="drop")),
                  ("scaler", preprocessing.StandardScaler())
                  ])
assert np.all(cs_t1.fit_transform(data) == ss_.fit_transform(data[cols_t1]))

# Pipelines with columns selector
clf = GridSearchCV(lm.LogisticRegression(class_weight='balanced'),
                    {'C': 10. ** np.arange(-3, 1)},
                    cv=cv_val, n_jobs=N_FOLDS_VAL)
 
clf_t0 = Pipeline([("selector", ColumnTransformer([("selector", "passthrough", cols_t0)], remainder="drop")),
                   ("scaler", preprocessing.StandardScaler()),
                   ("clf", clone(clf)),
                  ])

clf_t1 = Pipeline([("selector", ColumnTransformer([("selector", "passthrough", cols_t1)], remainder="drop")),
                   ("scaler", preprocessing.StandardScaler()),
                   ("clf", clone(clf)),
                   ])

clf_concat = Pipeline([("selector", ColumnTransformer([("selector", "passthrough", cols_t0 + cols_t1)], remainder="drop")),
                      ("scaler", preprocessing.StandardScaler()),
                      ("clf", clone(clf))
                      ])

clf_stack = StackingClassifier(
    estimators=[('t0', clf_t0), ('t1', clf_t1)], final_estimator=lm.LogisticRegression(class_weight='balanced'))

# Try and print architecture

clf_stack.fit(data, y)
clf_concat.fit(data, y)

# %% RUN
cv_res = cross_validate(estimator=clf_t0, X=data, y=y, cv=cv_train,
                        n_jobs=N_FOLDS,
                        scoring=['accuracy', 'balanced_accuracy',
                                'roc_auc'])

print("X0, CV ACC:%.3f±%.3f, bACC:%.3f±%.3f, AUC:%.3f±%.3f" % cross_validate_classif_summary(cv_res))
# X0, CV ACC:0.561±0.084, bACC:0.652±0.081, AUC:0.535±0.092
res.append(["X0", "lrl2_cv"] + list(cross_validate_classif_summary(cv_res)))

cv_res = cross_validate(estimator=clf_t1, X=data, y=y, cv=cv_train,
                        n_jobs=N_FOLDS,
                        scoring=['accuracy', 'balanced_accuracy',
                                'roc_auc'])

print("X1: CV ACC:%.3f±%.3f, bACC:%.3f±%.3f, AUC:%.3f±%.3f" % cross_validate_classif_summary(cv_res))
# X1: CV ACC:0.742±0.041, bACC:0.733±0.078, AUC:0.781±0.085
res.append(["X1", "lrl2_cv"] + list(cross_validate_classif_summary(cv_res)))

cv_res = cross_validate(estimator=clf_stack, X=data, y=y, cv=cv_train,
                        n_jobs=N_FOLDS,
                        scoring=['accuracy', 'balanced_accuracy',
                                'roc_auc'])
print("X01: CV ACC:%.3f±%.3f, bACC:%.3f±%.3f, AUC:%.3f±%.3f" % cross_validate_classif_summary(cv_res))
# X0X1: CV ACC:0.508±0.124, bACC:0.555±0.084, AUC:0.610±0.125
res.append(["X0X1", "lrl2_cv_stacked"] + list(cross_validate_classif_summary(cv_res)))

cv_res = cross_validate(estimator=clf_concat, X=data, y=y, cv=cv_train,
                        n_jobs=N_FOLDS,
                        scoring=['accuracy', 'balanced_accuracy',
                                'roc_auc'])
print("X0X1: CV ACC:%.3f±%.3f, bACC:%.3f±%.3f, AUC:%.3f±%.3f" % cross_validate_classif_summary(cv_res))
# X01: CV ACC:0.675±0.082, bACC:0.693±0.073, AUC:0.812±0.042
res.append(["X0X1", "lrl2_cv_concat"] + list(cross_validate_classif_summary(cv_res)))

# %% Save

res = pd.DataFrame(res, columns=["Data", "Model", "ACC-mean", "ACC-sd", "bACC-mean", "bACC-sd", "AUC-mean", "AUC-sd"])
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 2000)
print(res.round(3))

"""
    Data            Model  ACC-mean  ACC-sd  bACC-mean  bACC-sd  AUC-mean  AUC-sd
0     X0          lrl2_cv     0.561   0.084      0.652    0.081     0.535   0.092
1     X1          lrl2_cv     0.742   0.041      0.733    0.078     0.781   0.085
2     X0        lrenet_cv     0.542   0.131      0.500    0.000     0.500   0.000
3     X1        lrenet_cv     0.681   0.104      0.500    0.000     0.500   0.000
4     X0        svmrbf_cv     0.464   0.110      0.560    0.074     0.723   0.063
5     X1        svmrbf_cv     0.700   0.051      0.562    0.084     0.762   0.067
6     X0        forest_cv     0.700   0.036      0.633    0.056     0.618   0.118
7     X1        forest_cv     0.744   0.051      0.505    0.056     0.660   0.079
8     X0            gb_cv     0.725   0.066      0.457    0.038     0.542   0.062
9     X1            gb_cv     0.742   0.041      0.626    0.062     0.729   0.075
10    X0           mlp_cv     0.792   0.019      0.500    0.000     0.735   0.088
11    X1           mlp_cv     0.792   0.019      0.500    0.000     0.252   0.086
12    X0          lrl2_cv     0.561   0.084      0.652    0.081     0.535   0.092
13    X1          lrl2_cv     0.742   0.041      0.733    0.078     0.781   0.085
14  X0X1  lrl2_cv_stacked     0.508   0.124      0.555    0.084     0.610   0.125
15  X0X1   lrl2_cv_concat     0.675   0.082      0.693    0.073     0.812   0.042
"""
with pd.ExcelWriter(os.path.join(WD, "results", "res_classif.xlsx")) as writer:
    res.to_excel(writer, sheet_name='T0, T1 => psycho_PR', index=False)


""""
# %% Plot trajectories using PCA
# ==============================

from sklearn.decomposition import PCA

pca_t0 = PCA(n_components=1)
pca_t0.fit(data[cols_t0])
pca_t1 = PCA(n_components=1)
pca_t1.fit(data[cols_t1])

res_long =  pd.concat([
    pd.DataFrame(dict(PC1=pca_t0.transform(data[cols_t0]).squeeze(), y=y, t=0)),
    pd.DataFrame(dict(PC1=pca_t1.transform(data[cols_t1]).squeeze(), y=y, t=1))],
                      ignore_index=True, axis=0)

sns.violinplot(x="t", y="PC1",
             hue="y",
             data=res_long)

pca_t0.components_
pca_t1.components_

# %% Plot trajectories using concat linear model
# ==============================================

clf = clf_concat

bacc, auc = list(), list()
y_score_test_pred, y_score_test_true, y_score_test_pred_t0, y_score_test_pred_t1 = \
    np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))


for i, (train, test) in enumerate(cv_train.split(data, y)):
    print(f"Fold {i}:")
    clf.fit(data.iloc[train, :], y.iloc[train])
    y_score_test_pred[test] = clf.decision_function(data.iloc[test, :])
    y_score_test_true[test] = y.iloc[test]

    auc.append(metrics.roc_auc_score(y.iloc[test], y_score_test_pred[test]))
    bacc.append(metrics.balanced_accuracy_score(y.iloc[test], clf.predict(data.iloc[test, :])))

    # Split body head pipeline
    body, head_cv = pipeline_split_body_head(clf)
    # Check body + head = initial pipeline
    assert np.all(head_cv.decision_function(body.transform(data.iloc[test, :])) == clf.decision_function(data.iloc[test, :]))
    
    # Split linear head
    head = head_cv.best_estimator_
    # Check final coef size has expected shape
    assert head.coef_.shape[1] == len(cols_t0 + cols_t1)
    lm_t0, lm_t1 = linear_model_split(lm=head, where=len(cols_t0))

    Xtest_ = body.transform(data.iloc[test, :])
    y_score_test_pred_t0[test] = lm_t0.decision_function(Xtest_)
    y_score_test_pred_t1[test] = lm_t1.decision_function(Xtest_)
    assert np.allclose(y_score_test_pred_t0[test] + y_score_test_pred_t1[test] - head.intercept_,  y_score_test_pred[test])


assert np.all(y_score_test_true == y)
print("Test AUC:%.3f; bACC:%.3f" % (np.mean(auc), np.mean(bacc)))
# Test AUC:0.812; bACC:0.693
aucs_ = [metrics.roc_auc_score(y.iloc[test], y_score_test_pred[test]) for _, test in cv_train.split(data, y)]
np.mean(aucs_)

#y_score_test_pred_t0, y_score_test_pred_t1 = np.array(y_score_test_pred_t0), np.array(y_score_test_pred_t1)
metrics.roc_auc_score(y, y_score_test_pred_t0)
metrics.roc_auc_score(y, y_score_test_pred_t1)
metrics.roc_auc_score(y.values, y_score_test_pred)


# %% Plot


def logistic(x):
    return 1. / (1. + np.exp(-x))

import seaborn as sns

res = pd.DataFrame(dict(y_score_t0=y_score_test_pred_t0, y_prob_t0=logistic(y_score_test_pred_t0),
                        y_score_t1=y_score_test_pred_t1, y_prob_t1=logistic(y_score_test_pred_t1),
                        y=y.values))
# res = pd.DataFrame(dict(y_score_t0=np.log(y_score_t0), y_score_t1=np.log(y_score_t1), y=y))

sns.scatterplot(data=res, x='y_prob_t0', y='y_prob_t1', hue="y", alpha=0.5)
sns.pairplot(res[['y_prob_t0', 'y_prob_t1', 'y']], hue="y")




res_t0 = res[['y_prob_t0', 'y']]
res_t0["t"] = 0
res_t0 = res_t0.rename(columns={'y_prob_t0':'y_prob'})

res_t1 = res[['y_prob_t1', 'y']]
res_t1["t"] = 1
res_t1 = res_t1.rename(columns={'y_prob_t1':'y_prob'})


res_long =  pd.concat([res_t0, res_t1], ignore_index=True, axis=0)


sns.lineplot(x="t", y="y_prob",
             hue="y", style="y",
             data=res_long)

sns.lmplot(x="t", y="y_prob",
             hue="y",
             data=res_long)


sns.violinplot(x="t", y="y_prob",
             hue="y",
             data=res_long)

""""