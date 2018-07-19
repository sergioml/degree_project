"""
Experimento que hace un entrenamiento progresivo individual por cada mes, y el test es Mayo/2016
"""

import numpy as np
import pandas as pd
import time

import local_environment as local

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('../data/clean/train_clean.csv')
df_targets = pd.read_csv('../data/clean/train_labels.csv')

df_purcharsers = pd.read_csv('../data/clean/train_purcharsers.csv')
df_purcharsers.set_index('Unnamed: 0', inplace=True)

dates = df_purcharsers.fecha_dato.unique()
date_test = dates[-1]

results = pd.DataFrame(columns=['date', 'score', 'amount_data'])

for i, date in enumerate(dates[:-1]):
    df_x = df_purcharsers[df_purcharsers['fecha_dato'] == date]
    df_y = df_targets.loc[df_x.index]
    
    x_train = df_x.drop(['fecha_dato', 'fecha_alta'], axis=1).as_matrix()
    y_train = df_y.as_matrix()
    
    df_x_test = df[df['fecha_dato'] == date_test].drop(['fecha_dato', 'fecha_alta'], axis=1)
    df_y_test = df_targets.loc[df_x_test.index]
    
    x_test = df_x_test.as_matrix()
    y_test = df_y_test.as_matrix()
    
    model = local.model(x_train, y_train, DecisionTreeClassifier())
    probs, preds = local.calculatePredsProbs(x_test, model)
    
    df_prev = df[df['fecha_dato'] == dates[-2]]
    df_y = df_targets.loc[df_prev.index]
    
    predicted, actual = local.processPredictions(probs, preds, df_prev, df_x_test, df_y, df_y_test)
    
    score = local.mapk(actual, predicted, 7)
    
    results.loc[i] = [date, score, x_train.shape[0]]
    print(date)

results.to_csv('results/experiment6_DT_purcharsers.csv', index=False)

