from time import time

def submission(x_train, y_train, x_test, clf, target_cols, out_file, result_file, df_result, ncodpers_last_month):
    
    #x_train: array de entrenamiento - features
    #y_train: array de entrenamiento - targets
    #targets_cols: lista de los nombres de los targets
    #out_file: nombre del archivo txt de salida 
    #result_file: nombre del archivo csv de salida
    #df_result: dataframe para generar el archivo de salida - este DataFrame ya tiene los códigos de los clientes a predecir
    #ncodpers_last_month: serie de ncodpers que están en el mes anterior del test    
    
    y_df = y_train
    y_train = y_train.as_matrix()
    
    s = time()
    for i, col in enumerate(target_cols):
        f = open('results/txt/' + out_file + '.txt', 'a')
        start = time()
        clf.fit(x_train, y_train[:, i])
        preds = list(clf.predict(x_test))
        df_result[col] = preds
        end = time()
        f.write("{} {} {:0.4f} min\n".format(i, col, (end - start)/float(60)))
        f.close()

    rs = df_result.columns.tolist()[1:]

    for i in range(len(df_result)):
        
        ncodper = df_result.iloc[i]['ncodpers'] #Código individual
        
        if ncodper in ncodpers_last_month.values:
            ind = ncodpers_last_month[ncodpers_last_month == ncodper].index
            pre_targ = y_df.loc[ind].as_matrix()
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