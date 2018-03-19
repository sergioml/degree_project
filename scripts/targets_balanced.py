import numpy as np
import pandas as pd
from time import time

from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data/clean/train_clean_v2.csv')
df_targets = pd.read_csv('data/clean/train_labels.csv')

df_copy = df.copy()

x = df_copy.drop(['fecha_dato', 'fecha_alta', 'ncodpers'], axis=1).as_matrix()
y = df_targets.as_matrix()
print('Tamaño de datasets', x.shape, y.shape)

limit = 10094948 #Fila en la que inicia el último mes de registro

targets = df_targets.columns[:1]
for i, col in enumerate(targets):
    y_one_index = df_targets.iloc[y[:limit, i] == 1, i].index
    y_zero_index = df_targets.iloc[y[:limit, i] == 0, i].index

    n = np.min(np.unique(df_targets[col].as_matrix(), return_counts=True)[1])

    #Muestra de unos y ceros balanceados
    sample = np.sort(list(np.random.choice(y_one_index, size=n)) + list(np.random.choice(y_zero_index, size=n)))

    x_train = x[sample]
    y_train = y[sample, i]
    x_test = x[limit:]
    y_test = y[limit:, i]

    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    score = rf.score(x_test, y_test)

    print(col)
    print('El n de muestra es de', n, '--->', np.unique(df_targets[col].as_matrix(), return_counts=True))
    print('El score del entrenamiento es de', score)
    print('----------------------------------------')
