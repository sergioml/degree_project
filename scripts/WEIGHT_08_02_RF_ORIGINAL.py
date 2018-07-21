import numpy as np
import pandas as pd
import local_environment as local
from sklearn.ensemble import RandomForestClassifier
import time

df = pd.read_csv('../data/clean/train_clean.csv')
df_targets = pd.read_csv('../data/clean/train_labels.csv')

key = 'WEIGHT_08_02_RF_ORIGINAL'

print(key, '\n')
s = time.time()
f = open('final_experiments.txt', 'a')
f.write(key)
f.write('\nInicio' + time.strftime("%Y/%m/%d %H:%M") + '\n')
f.close()

df_a = df.loc[:, ['fecha_dato']].join(df_targets.loc[df.index])
df_a = df_a.groupby(['fecha_dato']).sum()
df_a.head()

dates = df_a.index

df_b = pd.DataFrame(columns=df_a.columns.tolist()[0:]) #dataframe products prev bought
for i in range(1, len(dates)):
    prev_prods = df_a.loc[df_a.index[i-1]].as_matrix()
    act_prods = df_a.loc[df_a.index[i]].as_matrix()
    bought_prods = act_prods - prev_prods
    df_b.loc[i] = bought_prods

df_a = df_a.reset_index().loc[:, ['fecha_dato']].join(df_b).iloc[1:]
df_a = df_a.set_index('fecha_dato')

dates = df_a.index[-6:]

results = pd.DataFrame(columns=['date_test', 'score', 'amount_data', 'time'])

for i in range(1, len(dates)):
    start = time.time()
    
    tuple_date = (dates[i-1], dates[i])
    
    f = open('final_experiments.txt', 'a')
    f.write(str(tuple_date[0]) + '-' + str(tuple_date[1]))
    f.write('\n')    
    f.close()
    
    weighted_class = []

    purchases = df_a.loc[tuple[0]]

    for p in purchases:
        if p > 0:
            weight_one = 0.8
            weight_zero = 1 - weight_one
            weighted_class.append({0: weight_zero, 1: weight_one})
        elif p < 0:
            weight_zero = 0.2
            weight_one = 1 - weight_zero
            weighted_class.append({0: weight_zero, 1: weight_one})
        else:
            weighted_class.append({0: 1, 1: 1})
           
    x_train = df.loc[df['fecha_dato'] == tuple_date[0]]
    y_train = df_targets.loc[x_train.index]

    x = x_train.drop(['fecha_dato', 'ncodpers','fecha_alta'], axis=1).as_matrix()
    y = y_train.as_matrix()
    
    model = local.model(x, y, RandomForestClassifier(n_jobs=4, class_weight=weighted_class))
    
    df_test = df.loc[df['fecha_dato'] == tuple_date[1]]
    df_test_targets = df_targets.loc[df_test.index]
    
    x_test = df_test.drop(['fecha_dato', 'ncodpers','fecha_alta'], axis=1).as_matrix()
    probs, preds = local.calculatePredsProbs(x_test, model)
    
    predicted, actual = local.processPredictions(probs, preds, x_train, df_test, y_train, df_test_targets)
    
    score = local.mapk(actual, predicted, 7)
    
    end = time.time()

    results.loc[i] = [tuple_date[1], score, x_train.shape[0], end-start]

results.to_csv('results/'+key+'.csv', index=False)
e = time.time()

f = open('final_experiments.txt', 'a')
f.write('\nFinal' + time.strftime("%Y/%m/%d %H:%M") + '\n')
final_time = (e - s)/3600
f.write(str(final_time)+'\n')
f.close()