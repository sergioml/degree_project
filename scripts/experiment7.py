"""
Experimento que hace un entrenamiento progresivo individual por cada mes, y el test es Mayo/2016
"""

import numpy as np
import pandas as pd
import time

import local_environment as local
import importlib

importlib.reload(local)

from sklearn.ensemble import RandomForestClassifier

s = time.time()

df = pd.read_csv('../data/clean/train_clean.csv')
df_targets = pd.read_csv('../data/clean/train_labels.csv')

dates = df.fecha_dato.unique()
date_test = dates[-1]

results = pd.DataFrame(columns=['date_start', 'date_end', 'score', 'amount_data'])

for i in range(1, len(dates[:-1])+1):
    date_range = dates[:i]
    df_x = df[df['fecha_dato'].isin(date_range)]
    df_y = df_targets.loc[df_x.index]
    
    x_train = df_x.drop(['fecha_dato', 'fecha_alta'], axis=1).as_matrix()
    y_train = df_y.as_matrix()
    
    df_x_test = df[df['fecha_dato'] == date_test].drop(['fecha_dato', 'fecha_alta'], axis=1)
    df_y_test = df_targets.loc[df_x_test.index]
    
    x_test = df_x_test.as_matrix()
    y_test = df_y_test.as_matrix()
    
    model = local.model(x_train, y_train, RandomForestClassifier(n_jobs=4))
    probs, preds = local.calculatePredsProbs(x_test, model)
    
    df_prev = df[df['fecha_dato'] == dates[-2]]
    df_y = df_targets.loc[df_prev.index]
    
    predicted, actual = local.processPredictions(probs, preds, df_prev, df_x_test, df_y, df_y_test)
    
    score = local.mapk(actual, predicted, 7)
    
    results.loc[i] = [date_range[0], date_range[-1], score, x_train.shape[0]]
    
results.to_csv('results/experiment7_v2.csv', index=False)

e = time.time()

print("Tiempo total {:.2f}".format((e - s)/3600))
