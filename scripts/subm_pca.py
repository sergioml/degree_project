import numpy as np
import pandas as pd
from time import time
from submission import submission

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

df = pd.read_csv('../data/clean/train_clean_v2.csv')
df_targets = pd.read_csv('../data/clean/train_labels.csv')
df_test = pd.read_csv('../data/clean/test_clean.csv')

df_copy = df.copy()
ncodpers_last_month = df_copy[df_copy.fecha_dato == df_copy.fecha_dato.value_counts().sort_index().index[-1]]['ncodpers']
df_copy.drop(['fecha_dato', 'fecha_alta', 'ncodpers'], axis=1, inplace=True)

resultados = pd.DataFrame(columns= ['ncodpers'] + df_targets.columns.tolist() + ['added_products'])
resultados['ncodpers'] = df_test['ncodpers']

df_test.drop(['fecha_dato', 'fecha_alta', 'ncodpers', 'ind_nuevo'], axis=1, inplace=True)

x = df_copy.as_matrix()
y = df_targets
x_test = df_test.as_matrix()

PCA = PCA(n_components=10)
x_train_pca = PCA.fit_transform(x)
x_test_pca = PCA.fit_transform(x_test)
columns = df_targets.columns.tolist()

clf = RandomForestClassifier()

submission(x_train=x_train_pca,
        y_train=y,
        x_test=x_test_pca,
        target_cols=columns,
        clf = clf,
        out_file='out_pca_n10_prevprods_v2',
        result_file='resultados_pca_n10_prevprods_v2',
        df_result=resultados,
        ncodpers_last_month=ncodpers_last_month)