import numpy as np
import pandas as pd
import time

import local_environment as local

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('../data/clean/train_clean.csv')
df_targets = pd.read_csv('../data/clean/train_labels.csv')
df_purchasers = pd.read_csv('../data/clean/train_purchasers.csv', index_col='index')

key = 'PREV_PRODS_RF_PURCHASERS'
print(key)

s = time.time()

f = open('final.txt', 'a')
f.write('\n' + key)
f.write('\nInicio ' + time.strftime("%Y/%m/%d %H:%M") + '\n')
f.close()

dates = df['fecha_dato'].unique()
date_test = dates[-6:]

results = pd.DataFrame(columns=['date_test', 'score', 'amount_data', 'time'])

for i in range(1, len(date_test)):
    
    start = time.time()
    
    f = open('final.txt', 'a')
    f.write(str(date_test[i-1]) + ' - ' + str(date_test[i]))
    f.write('\n')    
    f.close()
    
    
    date_range = dates[:-(len(date_test)-i)]
    
    #Procesamiento de datos de entrenamiento
    
    df_aux = df_purchasers.loc[df_purchasers['fecha_dato'].isin(date_range[:-1]), ['ncodpers']].join(df_targets.loc[df_purchasers.index])    
    df_aux = df_aux.groupby(['ncodpers']).sum()
    df_aux.reset_index(inplace=True)
    
    x_train = df_purchasers.loc[df_purchasers['fecha_dato'] == date_test[i-1]]
    y_train = df_targets.loc[x_train.index].reset_index(drop=True)
    x_train = x_train.merge(df_aux, on='ncodpers', how='left')
    x_train.replace(np.nan, 0, inplace=True)
    
    x = x_train.drop(['fecha_dato', 'fecha_alta'], axis=1).as_matrix()
    y = y_train.as_matrix()
    
    model = local.model(x, y, RandomForestClassifier(n_jobs=4))
    
    #Procesamiento de datos de test
    
    df_aux = df_purchasers.loc[df_purchasers['fecha_dato'].isin(date_range), ['ncodpers']].join(df_targets)
    df_aux = df_aux.groupby(['ncodpers']).sum()
    df_aux.reset_index(inplace=True)
    
    df_test = df.loc[df['fecha_dato'] == date_test[i]]
    y_test = df_targets.loc[df_test.index]
    
    x_test = df_test.merge(df_aux, on='ncodpers', how='left')
    x_test.replace(np.nan, 0, inplace=True)
    x_test = x_test.drop(['fecha_dato', 'fecha_alta'], axis=1).as_matrix()
    
    probs, preds = local.calculatePredsProbs(x_test, model)
    
    #Validación del modelo
    
    x_prev = df.loc[df['fecha_dato'] == date_test[i-1]]
    y_prev = df_targets.loc[x_prev.index]
    
    predicted, actual = local.processPredictions(probs, preds, x_prev, df_test, y_prev, y_test)
    
    score = local.mapk(actual, predicted, 7)
    
    end = time.time()
    
    results.loc[i] = [date_test[i], score, x_train.shape[0], end-start]

results.to_csv('results/'+key+'.csv', index=False)
e = time.time()
f = open('final.txt', 'a')
f.write('\nFinal ' + time.strftime("%Y/%m/%d %H:%M") + '\n')
final_time = (e - s)/60
f.write(str(final_time)+'\n')
f.close()
    

