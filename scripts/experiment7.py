"""
Experimento que hace un entrenamiento progresivo individual por cada mes, y el test es Mayo/2016
"""

import numpy as np
import pandas as pd
import time
import sys

import local_environment as local

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

s = time.time()

df = pd.read_csv('../data/clean/train_clean.csv')
df_targets = pd.read_csv('../data/clean/train_labels.csv')

df_purchasers = pd.read_csv('../data/clean/train_purchasers.csv', index_col='index')

df_train = df_purchasers.copy()
k = -5
dates = df_train.fecha_dato.unique()
date_test = dates[k]
print('Datasets cargados')
print(df_train.shape)
results = pd.DataFrame(columns=['date_start', 'date_end', 'score', 'amount_data'])
print('Inicio')
for i in range(1, len(dates[:k])+1):
    date_range = dates[:i]
    df_x = df_train[df_train['fecha_dato'].isin(date_range)]
    df_y = df_targets.loc[df_x.index]
    
    x_train = df_x.drop(['fecha_dato', 'fecha_alta'], axis=1).as_matrix()
    y_train = df_y.as_matrix()
    
    df_x_test = df[df['fecha_dato'] == date_test].drop(['fecha_dato', 'fecha_alta'], axis=1)
    df_y_test = df_targets.loc[df_x_test.index]
    
    x_test = df_x_test.as_matrix()
    y_test = df_y_test.as_matrix()
    
    model = local.model(x_train, y_train, RandomForestClassifier(n_jobs=4))
    probs, preds = local.calculatePredsProbs(x_test, model)
    
    df_prev = df[df['fecha_dato'] == dates[k-1]]
    df_y = df_targets.loc[df_prev.index]
    
    predicted, actual = local.processPredictions(probs, preds, df_prev, df_x_test, df_y, df_y_test)
    
    score = local.mapk(actual, predicted, 7)
    
    results.loc[i] = [date_range[0], date_range[-1], score, x_train.shape[0]]
    print("({} - {}, score {})".format(date_range[0], date_range[-1], score))
    

#key = sys.argv[1] #Key para todos los experimentos iguales
#dataset = sys.argv[2]
key = "RF_ENERO16"
dataset = "PURCHASERS"
name_file = 'experiment7_'+key+'_'+dataset+'.csv'
print(name_file)
results.to_csv('results/'+name_file, index=False)

e = time.time()

print("Tiempo total {:.2f}".format((e - s)/3600))
