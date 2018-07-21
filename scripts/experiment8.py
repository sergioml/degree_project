import numpy as np
import pandas as pd

import local_environment as local

from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('../data/clean/train_clean.csv')
df_targets = pd.read_csv('../data/clean/train_labels.csv')

df_test = pd.read_csv('../data/clean/test_clean.csv')

x_train = df.copy()
y_train = df_targets.loc[x_train.index]

x = x_train.drop(['fecha_dato', 'fecha_alta', 'ncodpers'], axis=1)
y = y_train.as_matrix()

model = local.model(x, y, DecisionTreeClassifier())

x_test = df_test.drop(['fecha_dato', 'fecha_alta', 'ncodpers'], axis=1).as_matrix()
probs, preds = local.calculatePredsProbs(x_test, model)

x_prev = df.loc[df['fecha_dato'] == '2016-05-28']
subm = local.processPredictions(probs=probs, preds=preds, df_prev=x_prev, df_test=df_test, df_targets=y_train, env='submit', path='../results/submissions/')
