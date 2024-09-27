import os
import pandas as pd
import numpy as np

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

# %% Load data

data_t0 = pd.read_excel(os.path.join(WD, "data", sourcedata_filename), sheet_name='DataT0', skiprows=2)
data_t1 = pd.read_excel(os.path.join(WD, "data", sourcedata_filename), sheet_name='DataT1', skiprows=2)
data_t2 = pd.read_excel(os.path.join(WD, "data", sourcedata_filename), sheet_name='DataT2', skiprows=2)

data_t0 = pd.concat([
    data_t0['participant_id'],
    data_t0[cols_t0_gradNeuroDev].sum(axis=1),
    data_t0[cols_t0_DSM].sum(axis=1),
    data_t0[cols_t0_DSM]],
          axis=1)

print(data_t0.shape)
data_t0.columns = ['participant_id'] + cols_t0


assert data_t0.shape == (49, 9)

data_t1 = data_t1[['participant_id'] + cols_t1]
assert data_t1.shape == (49, 7)

data_t2 = data_t2[['participant_id'] + cols_t2].dropna()
assert data_t2.shape == (43, 6)

data = pd.merge(left=pd.merge(left=data_t0, right=data_t1, on='participant_id'),
                right=data_t2, on='participant_id',  how='outer')
assert data.shape == (49, 20)

data.to_csv(os.path.join(WD, "data", data_filename), index=False)

# %%
