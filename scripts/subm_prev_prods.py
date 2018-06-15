#En este script se tienen en cuenta los productos del mes inmediatamente anterior

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from submission import submission

df = pd.read_csv('../data/clean/train_clean_v2.csv')
df_targets = pd.read_csv('../data/clean/train_labels.csv')
df_test = pd.read_csv('../data/clean/test_clean.csv')
df_copy = df.copy()

#### Procesamiento de dataset de train

groupby_fecha_dato = df_copy.groupby(['fecha_dato'])

dates = df_copy.fecha_dato.unique()

df_new = pd.DataFrame(columns=df_copy.columns.tolist() + df_targets.columns.tolist())
for i in range(1, len(dates)):
    pre = groupby_fecha_dato.get_group(dates[i-1]) #mes anterior
    pos = groupby_fecha_dato.get_group(dates[i]) #mes actual
    
    codes_in_pre_and_pos = list(set(pos.ncodpers.values) & set(pre.ncodpers.values)) #clientes que están en ambos meses
    
    df_aux = df_targets.loc[pre.ncodpers.loc[pre.ncodpers.isin(codes_in_pre_and_pos)].index]
    df_aux = df_aux.join(df_copy.ncodpers, how='inner')
    df_aux = df_aux[[df_aux.columns.tolist()[-1]] + df_aux.columns.tolist()[:-1]]
    
    pos = pos.merge(df_aux, on='ncodpers', how='outer', right_index=True)
    pos.fillna(0, inplace=True)
    df_new = df_new.append(pos)


x_train = df_new.drop(['fecha_dato', 'fecha_alta', 'ncodpers'], axis=1).as_matrix()
y_train = df_targets.loc[df_new.index]

#### Dataset test ####

df_new_test = pd.DataFrame(columns=df_copy.columns.tolist() + df_targets.columns.tolist())
pre = groupby_fecha_dato.get_group(dates[-1]) #mes anterior test - ultimo mes de entrenamiento
pos = df_test.drop(['ind_nuevo'], axis=1) #mes test

codes_in_pre_and_pos = list(set(pos.ncodpers.values) & set(pre.ncodpers.values)) #clientes que están en ambos meses

df_aux = df_targets.loc[pre.ncodpers.loc[pre.ncodpers.isin(codes_in_pre_and_pos)].index]
df_aux = df_aux.join(df_copy.ncodpers, how='inner')
df_aux = df_aux[[df_aux.columns.tolist()[-1]] + df_aux.columns.tolist()[:-1]]

pos = pos.merge(df_aux, on='ncodpers', how='outer', right_index=True)
pos.fillna(0, inplace=True)
df_new_test = df_new_test.append(pos)

x_test = df_new_test.drop(['fecha_dato', 'fecha_alta', 'ncodpers'], axis=1).as_matrix()

###------###

clf = RandomForestClassifier()
target_cols = df_targets.columns.tolist()
out_file = 'prev_prods'
result_file = 'only_prev_prods'

df_result = pd.DataFrame(columns= ['ncodpers'] + df_targets.columns.tolist() + ['added_products'])
df_result['ncodpers'] = df_test['ncodpers']

ncodpers_last_month = pre['ncodpers']

submission(x_train, y_train, x_test, clf, target_cols, out_file, result_file, df_result, ncodpers_last_month)







