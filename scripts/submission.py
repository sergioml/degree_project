import numpy as np
import pandas as pd
from time import time

from sklearn.ensemble import RandomForestClassifier

df_data = pd.read_csv('../data/clean/train_clean_v2.csv')
df_targets = pd.read_csv('../data/clean/train_labels.csv')

df_copy = df_data.copy()

df_test = pd.read_csv('../data/clean/test_clean.csv')

##Datos de entrenamiento
df_copy.drop(['fecha_dato', 'fecha_alta', 'ncodpers'], axis=1, inplace=True)
x_train = df_copy.as_matrix()
y_train = df_targets.as_matrix()

resultados = pd.DataFrame(columns= ['ncodpers'] + df_targets.columns.tolist() + ['added_products'])
resultados['ncodpers'] = df_test['ncodpers']

df_test.drop(['fecha_dato', 'fecha_alta', 'ncodpers'], axis=1, inplace=True)
x_test = df_test.as_matrix()

rf = RandomForestClassifier()

for i, col in enumerate(df_targets.columns.tolist()):
    rf.fit(x_train, y_train[:, i])
    preds = [rf.predict(x.reshape(1, -1))[0] for x in x_test]
    resultados[col] = preds
rs = resultados.columns.tolist()[1:]
for i in range(len(resultados)):
    line = resultados.iloc[i, 1:-1].as_matrix()
    resultados.iloc[i , -1] = " ".join([rs[i] for i, t in enumerate(line) if t == 1])

resultados.to_csv('resultados_script.csv', index=False)
