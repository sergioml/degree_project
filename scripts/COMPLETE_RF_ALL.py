import numpy as np
import pandas as pd
import local_environment as local
from sklearn.ensemble import RandomForestClassifier
import time

df = pd.read_csv('../data/clean/train_clean.csv')
df_targets = pd.read_csv('../data/clean/train_labels.csv')
df_purchasers = pd.read_csv('../data/clean/train_purchasers.csv', index_col='index')

key = 'COMPLETE_RF_PURCHASERS'

print(key, '\n')
s = time.time()
f = open('final_experiments.txt', 'a')
f.write('\n' + key)
f.write('\nInicio ' + time.strftime("%Y/%m/%d %H:%M") + '\n')
f.close()

dates = df.loc[:, 'fecha_dato'].unique()

results = pd.DataFrame(columns=['date_test', 'score', 'amount_data', 'time'])

for i in range(5):
    start = time.time()

    training_dates = dates[:12+i]
    tuple_date = dates[[i+11,i+12]]
    
    f = open('final_experiments.txt', 'a')
    f.write(str(tuple_date[0]) + ' - ' + str(tuple_date[1]))
    f.write('\n')    
    f.close()
    
    x_train = df.loc[df['fecha_dato'].isin(training_dates)]
    y_train = df_targets.loc[x_train.index]

    x = x_train.drop(['fecha_dato', 'ncodpers','fecha_alta'], axis=1).as_matrix()
    y = y_train.as_matrix()
    
    model = local.model(x, y, RandomForestClassifier(n_jobs=4))
    
    df_test = df.loc[df['fecha_dato'] == tuple_date[1]]
    df_test_targets = df_targets.loc[df_test.index]
    
    x_test = df_test.drop(['fecha_dato', 'ncodpers','fecha_alta'], axis=1).as_matrix()
    probs, preds = local.calculatePredsProbs(x_test, model)
    
    x_prev = df.loc[df['fecha_dato'] == tuple_date[0]]
    y_prev = df_targets.loc[x_prev.index]
    
    predicted, actual = local.processPredictions(probs, preds, x_prev, df_test, y_prev, df_test_targets)
   
    score = local.mapk(actual, predicted, 7)
    
    end = time.time()

    results.loc[i] = [tuple_date[1], score, x_train.shape[0], end-start]

results.to_csv('results/last/'+key+'.csv', index=False)
e = time.time()
f = open('final_experiments.txt', 'a')
f.write('\nFinal ' + time.strftime("%Y/%m/%d %H:%M") + '\n')
final_time = (e - s)/3600
f.write(str(final_time)+'\n')
f.close()