import numpy as np
import pandas as pd
from time import time

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
y = df_targets.as_matrix()
x_test = df_test.as_matrix()

def results(x_train, y_train, x_test, target_cols, out_file, result_file, df_result, ncodpers_last_month):
    
    #x_train: array de entrenamiento - features
    #y_train: array de entrenamiento - targets
    #targets_cols: lista de los nombres de los targets
    #out_file: nombre del archivo txt de salida 
    #result_file: nombre del archivo csv de salida
    #df_result: dataframe para generar el archivo de salida - este DataFrame ya tiene los códigos de los clientes a predecir
    #ncodpers_last_month: serie de ncodpers que están en el mes anterior del test    
    
    s = time()
    for i, col in enumerate(target_cols):
        f = open('results/txt/' + out_file + '.txt', 'a')
        start = time()
        rf = RandomForestClassifier(n_jobs=4)
        rf.fit(x_train, y_train[:, i])
        preds = list(rf.predict(x_test))
        df_result[col] = preds
        end = time()
        f.write("{} {} {:0.4f} min\n".format(i, col, (end - start)/float(60)))
        f.close()

    rs = df_result.columns.tolist()[1:]

    for i in range(len(df_result)):
        
        ncodper = df_result.iloc[i]['ncodpers'] #Código individual
        
        if ncodper in ncodpers_last_month.values:
            pre_targ = y_train[ncodpers_last_month[ncodpers_last_month == ncodper].index]
            post_targ = df_result.iloc[i, 1:-1].as_matrix()
            line = (pre_targ - post_targ).reshape((24))
            df_result.iloc[i , -1] = " ".join([rs[i] for i, t in enumerate(line) if t == -1])
        else:
            line = df_result.iloc[i, 1:-1].as_matrix()
            df_result.iloc[i , -1] = " ".join([rs[i] for i, t in enumerate(line) if t == 1])

    e = time()
    f = open('results/txt/' + out_file + '.txt', 'a')
    f.write("\nTiempo total: {:0.4f} min".format((e-s)/float(60)))
    f.close()
    df_result[['ncodpers', 'added_products']].to_csv('results/submissions/' + result_file + '.csv', index=False)
        
#--------#
PCA = PCA(n_components=10)
x_train_pca = PCA.fit_transform(x)
x_test_pca = PCA.fit_transform(x_test)
columns = df_targets.columns.tolist()

results(x_train=x_train_pca,
        y_train=y,
        x_test=x_test_pca,
        target_cols=columns,
        out_file='out_pca_n10_prevprods',
        result_file='resultados_pca_n10_prevprods',
        df_result=resultados,
        ncodpers_last_month=ncodpers_last_month)