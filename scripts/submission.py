from time import time
import pandas as pd

def submission(x_train, y_train, x_test, clf, out_file, result_file, ncodpers_last_month):
"""    
    * x_train: DataFrame
            Datos de entrenamiento - features 
    * y_train: DataFrame
            Targets de entrenamiento
    * x_test: DataFrame
            Datos de test con preprocesado con NCODPERS
    * clf: Modelo de entrenamiento - RandomForestClassifier, etc.
    * out_file: str
            nombre del archivo txt de salida
    * result_file: str 
            nombre del archivo csv de salida
    * ncodpers_last_month: Serie
            ncodpers que están en el mes anterior del test    
"""    
    y_df = y_train
    y_train = y_train.as_matrix()
    x_tr = x_train.as_matrix()
    x_t = x_test.drop(['ncodpers']).as_matrix()
    
    target_cols = y_df.columns.tolist()
    
    #Creación de DataFrame de resultados de predicciones
    cols = [sorted(target_cols*2), ['pred', 'prob']*(len(target_cols))]
    tuples = list(zip(*cols))
    index = pd.MultiIndex.from_tuples(tuples)
    result = pd.DataFrame(columns=index)
    result.insert(0, 'ncodpers', x_test.ncodpers.values)

    s = time()
    for i, col in enumerate(target_cols):
        f = open('results/txt/' + out_file + '.txt', 'a')
        start = time()
        clf.fit(x_tr, y_train[:, i])
        
        pred_probs = rf.predict_proba(x_t).max(axis=1)
        preds = rf.predict(x_t)
        
        result[col] = np.vstack((preds, pred_probs)).T
        end = time()
        f.write("{} {} {:0.4f} min\n".format(i, col, (end - start)/float(60)))
        f.close()

    #Creción del archivo de submission
    submission = pd.DataFrame(columns=['ncodpers'] + target_cols + ['added_products'])
    submission['ncodpers'] = result['ncodpers']
    
    for i in target_cols:
        submission[i] = list(zip(*[[i]*len(result), result[i]['pred'] * result[i]['prob']]))

    for i in range(len(result)):
        
        ncodper = result.iloc[i]['ncodpers'] #Código individual
        
        if ncodper in ncodpers_last_month.values:
            ind = ncodpers_last_month[ncodpers_last_month == ncodper].index
            pre_targ = y_df.loc[ind].as_matrix().reshape(24)
            
            predicted_targ = result[result.ncodpers == ncodper].loc[:, tuples].as_matrix().reshape(48)
            predicted_targ = np.array([predicted_targ[i] for i in range(0, predicted_targ.shape[0], 2)])
            
            line = (pre_targ - predicted_targ)
            
            #predicted_prods son las predicciones de los productos con los 'labels'
            predicted_prods = submission[submission.ncodpers == ncodper].loc[:, target_cols].values.reshape(24)         
            
            final_prediction = np.array([predicted_prods[i] for i, p in enumerate(line) if p != 1])
            predicted_str = " ".join([j[0] for j in sorted(final_prediction, key=lambda x: x[1], reverse=True)][:7])
            
        else:
            predicted_prods = submission[submission.ncodpers == ncodper].loc[:, target_cols].values.reshape(24)
            predicted_str = " ".join([j[0] for j in sorted(predicted_prods, key=lambda x: x[1], reverse=True)][:7])
    
        result.iloc[i , -1] = predicted_str

    e = time()
    f = open('results/txt/' + out_file + '.txt', 'a')
    f.write("\nTiempo total: {:0.4f} min".format((e-s)/float(60)))
    f.close()
    result[['ncodpers', 'added_products']].to_csv('results/submissions/' + result_file + '.csv', index=False)