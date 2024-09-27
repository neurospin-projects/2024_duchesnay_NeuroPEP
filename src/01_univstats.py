import os
import pandas as pd
import numpy as np

import itertools
import scipy

import matplotlib.pyplot as plt
import seaborn as sns

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig_w, fig_h = plt.rcParams.get('figure.figsize')
# plt.rcParams['figure.figsize'] = (fig_w, fig_h * .5)

# %% config
# =========

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
# ========

def univ_stats(data, col1, col2):

    res = list()

    for x, y in itertools.product(col1, col2):
        df_ = data[[x, y]].dropna()
        
        if np.all(np.unique(df_.values) == [0, 1]): # Chi2
            crosstab = pd.crosstab(df_[x], df_[y], rownames=[x], colnames=[y])
            stat, pval, dof, expected = scipy.stats.chi2_contingency(crosstab)
            test = "chi2"
            
        elif np.all(np.unique(df_[y].values) == [0, 1]): # two-sample t-test / y
            ttest = scipy.stats.ttest_ind(df_.loc[df_[y] == 1, x], df_.loc[df_[y] == 0, x], equal_var=False)
            stat, pval = ttest.statistic, ttest.pvalue
            test = "ttest"
        
        elif np.all(np.unique(df_[x].values) == [0, 1]): # two-sample t-test / x
            ttest = scipy.stats.ttest_ind(df_.loc[df_[x] == 1, y], df_.loc[df_[x] == 0, y], equal_var=False)
            stat, pval = ttest.statistic, ttest.pvalue
            test = "ttest"
        
        else:
            test = scipy.stats.pearsonr(df_[x], df_[y])
            stat, pval = test.statistic, test.pvalue
            test = "corr"
        
        res.append([x, y, stat, pval, test])

    res = pd.DataFrame(res, columns=['v1', 'v2', 'stat', 'pval', 'test'])
    res = res.sort_values( 'pval')
    return(res)


def box_plot(*args, color, lw=1, **kwargs):
    """Boxplot helper
    
    https://matplotlib.org/stable/gallery/statistics/boxplot.html#sphx-glr-gallery-statistics-boxplot-py
    
    medians: horizontal lines at the median of each box.
    whiskers: the vertical lines extending to the most extreme, non-outlier data points.
    caps: the horizontal lines at the ends of the whiskers.
    fliers: points representing data that extend beyond the whiskers (fliers).
    means: points or lines representing the means.
    """
    boxprops = dict(linestyle='-', lw=lw, color=color) # The style of the box.
    whiskersprops= dict(linestyle='-', lw=lw, color=color) # The style of the box.
    capssprops= dict(linestyle='-', lw=lw, color=color) # The style of the box.
    meanprops= dict(linestyle='-', lw=lw, marker='o', mec=color, mfc=color, color=color) # The style of the box.
    medianprops= dict(linestyle='-', lw=lw, color=color) # The style of the box.

    plt.boxplot(*args, **kwargs, showmeans=True, patch_artist=False, showfliers=False,
                boxprops=boxprops,
                capprops=capssprops,
                whiskerprops=whiskersprops,
                #flierprops=dict(color=c, markeredgecolor=c),
                medianprops=medianprops,
                meanprops=meanprops,
                )


# %% Load data
# ============

data = pd.read_csv(os.path.join(WD, "data", data_filename))

# %% Descriptive statistics
# =========================

# type des variables

# T0 Quanti + 0, 1
print(data[cols_t0].head())
# gradNeuroDev  DSM_Sum => Quanti
# le reste [0, 1]

# T1 Quanti
print(data[cols_t1].head())
# quanti sauf Catatonie

# T2 [0, 1]
print(data[cols_t2].head())
# cols_t1 : quanti sauf Catatonie

# Matrice de correlation
corr = data.drop("participant_id", axis=1).corr()


plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(corr, vmin=-1, vmax=1, annot=True, cmap='coolwarm')
#corr.style.background_gradient(cmap='coolwarm')
heatmap.set_title('Correlation matrix', fontdict={'fontsize':12}, pad=12);
#plt.show()
plt.savefig(os.path.join(WD, "results", "corr.png"), dpi=300, bbox_inches='tight')
# plt.savefig(os.path.join(WD, "results", "corr.pdf"), dpi=300, bbox_inches='tight')
plt.close()


# %% Univariate Statistics
# ========================

# Influence de 3 variables T0 sur les symptômes cliniques de nos patients à T1 (6 variables)

rest_stat_t0t1 = univ_stats(data=data, col1=cols_t0, col2=cols_t1)
rest_stat_t0t1.head()


# Influence de 3 variables T0 sur les symptômes persistants à T2 (5 variables).

rest_stat_t0t2 = univ_stats(data=data, col1=cols_t0, col2=cols_t2)
rest_stat_t0t2.head()

# T1 sur les symptômes persistants à T2 (5 variables).

rest_stat_t1t2 = univ_stats(data=data, col1=cols_t1, col2=cols_t2)
rest_stat_t1t2.head()

# Save

with pd.ExcelWriter(os.path.join(WD, "results", "res_stat.xlsx")) as writer:
    corr.to_excel(writer, sheet_name='corr')
    rest_stat_t0t1.to_excel(writer, sheet_name='t0t1', index=False)
    rest_stat_t0t2.to_excel(writer, sheet_name='t0t2', index=False)
    rest_stat_t1t2.to_excel(writer, sheet_name='t1t2', index=False)


# %% Plot trajectories using feture from univariate statistics
# ============================================================

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


cols_t0_ = ['DSM_Sum', 'gradNeuroDev']
cols_t0_ = ['DSM_Sum']

cols_t1_ = ["CAARMS4s", "CAARMS2s", "CAARMS1s"]
col_t2_ = ['psycho_PR']
data_ = data[cols_t0_ + cols_t1_ + col_t2_]

data_ = data_[data_["psycho_PR"].notnull()]
[[x, data_[x].isnull().sum()] for x in data_.columns]
y = data_['psycho_PR']

# 1. Descriptives stat

corr = data_[cols_t0_ + cols_t1_].corr()
print(corr.round(3))

print(data_[cols_t0_ + cols_t1_].var().round(3))

"""
Corr
              DSM_Sum  gradNeuroDev  CAARMS4s  CAARMS2s  CAARMS1s
DSM_Sum         1.000         0.609     0.043     0.263    -0.240
gradNeuroDev    0.609         1.000    -0.003     0.152    -0.061
CAARMS4s        0.043        -0.003     1.000     0.231     0.035
CAARMS2s        0.263         0.152     0.231     1.000     0.027
CAARMS1s       -0.240        -0.061     0.035     0.027     1.000

Var
DSM_Sum          1.642
gradNeuroDev    15.344
CAARMS4s         1.359
CAARMS2s         1.754
CAARMS1s         0.635
dtype: float64
"""

# 2. Summarize scores

ss = StandardScaler()

"""
pca_t0 = PCA(n_components=1).fit(ss.fit_transform(data_[cols_t0_]))
if np.all(np.sign(pca_t0.components_) == -1):
    pca_t0.components_ *= -1
score_t0 = pca_t0.transform(ss.fit_transform(data_[cols_t0_])).squeeze()
"""
score_t0 = ss.fit_transform(data_[cols_t0_]).squeeze()

pca_t1 = PCA(n_components=1).fit(ss.fit_transform(data_[cols_t1_]))
if np.all(np.sign(pca_t1.components_) == -1):
    pca_t1.components_ *= -1
score_t1 = pca_t1.transform(ss.fit_transform(data_[cols_t1_])).squeeze()


score_t0.shape, score_t1.shape, y.shape, data_.shape

res_long =  pd.concat([
    pd.DataFrame(dict(score=score_t0, y=y, t=0)),
    pd.DataFrame(dict(score=score_t1, y=y, t=1))],
                      ignore_index=True, axis=0)


# 2. Box plot

widths = .1
x = np.array([0, .5])
no_x = x - widths
yes_x = x + widths
xlim = np.min(x) - 2* widths, np.max(x) + 2 * widths

fig_w, fig_h
fig, ax = plt.subplots()
#fig.set_size_inches(fig_w/2, fig_h) 

box_plot([score_t0[y==0], score_t1[y==0]], positions=no_x, lw=1, color=colors[0], widths=.1)
box_plot([score_t0[y==1], score_t1[y==1]], positions=yes_x, lw=1, color=colors[1], widths=.1)

plt.plot(no_x, [np.mean(score_t0[y==0]), np.mean(score_t1[y==0])], color=colors[0], lw=2, label="No psycho_PR")
plt.plot(yes_x, [np.mean(score_t0[y==1]), np.mean(score_t1[y==1])], color=colors[1], lw=2, label="psycho_PR")
plt.legend()

ax.set_xticks(x, ["T0(%s)" % ", ".join(cols_t0_), "T1(%s)" % ", ".join(cols_t1_)])
ax.set_xlim(xlim)
ax.set_xlabel('Time')
ax.set_ylabel('Standardize score or PC1')

#plt.savefig(os.path.join(WD, "results", "trajectories_T0-T1_%s.png" % col_t2_[0]), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(WD, "results", "trajectories_T0-T1_%s.svg" % col_t2_[0]), dpi=300, bbox_inches='tight')

plt.close()



# [Momentum Gradient Descent](https://www.youtube.com/watch?v=iudXf5n_3ro)
# [Gradient Descent](https://www.youtube.com/watch?v=qg4PchTECck)

# %%
