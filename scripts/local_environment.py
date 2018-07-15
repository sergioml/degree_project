import numpy as np
import pandas as pd
import time

def model(x_train, y_train, model):
    """
    Parameters
    ----------
    x_train: Array
        Datos de entrenamiento previamente procesados
    y_train: Array
        Targets de entrenamiento
    model: Objeto (de sklearn)
        Algoritmo para entrenar con los paramétros configurados
    
    Returns
    -------
    model: Objeto
        Algortimo entrenado
    """
    return model.fit(x_train, y_train)

def calculatePredsProbs(x_test, clf):
    """
    Función para calcular las probabilidades de predicción y predicciones hechas por el modelo entrenado
    
    Parameters
    ----------
    x_test: Array
        Datos de test
    clf: Objeto
        El modelo previamente entrenado
        
    Results
    -------
    probs: Array
        Array de las probabilidades de predicción
    preds: Array
        Array de predicciones

    """
    preds = clf.predict(x_test)
    
    probs = clf.predict_proba(x_test)
    probs = np.array([pr.max(axis=1) for pr in probs]).T
    
    return probs, preds

def processPredictions(probs=None, preds=None, df_prev=None, df_test=None,
                       df_targets=None, y_test=None, env='local', path='results/submissions/'):
    """
    Procesamiento de las predicciones hechas para generar el archivo de submission
    o los datos necesarios para hacer una validación local
    Parameters
    ----------
    probs: Array
        Probabilidades de elegir un producto - Generado por el modelo
    preds: Array
        Predicciones hechas por el modelo entrenado
    df_prev: DataFrame
        Datos del mes previo al de test
    df_test: DataFrame
        Datos del mes de test
    df_targets: DataFrame
        Targets del mes previo del mes de entrenamiento
    y_test: DataFrame
        Targets del mes de test, sólo si env es 'local'
    env: str (optional)
        Indica el tipo de ejecución que se quiere hacer
            'local': regresa dos listas para hacer la validación local - Default
            'submit': regresa un DataFrame con el archivo de submission csv (el archivo de submission se genera
                    y se guarda en el equipo)
    path: str (optional)
        Dirección de la carpeta donde se va a guardar el archivo
    
    Returns
    -------
    Si env es 'submit'
        df_subm: DataFrame
            DataFrame con el archivo de submission csv (el archivo de submission se genera y se guarda en el equipo)
    Si env es 'local'
        predicted: list
            Una lista de listas de los productos que se predijeron
    
    """
    
    df_prev.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    df_targets.reset_index(drop=True, inplace=True)
    
    ncodpers_prev_month = df_prev.loc[:, 'ncodpers'].values
    ncodpers_last_month = df_test.loc[:, 'ncodpers'].values
    
    ncodpers_both = list(set(ncodpers_last_month) & set(ncodpers_prev_month))
    
    index_prev = df_prev[df_prev['ncodpers'].isin(ncodpers_both)].sort_values(['ncodpers']).index
    index_last = df_test[df_test['ncodpers'].isin(ncodpers_both)].sort_values(['ncodpers']).index
    
    prev_prods = df_targets.loc[index_prev].as_matrix()
    pred_prods = preds[index_last, :]
    
    #Predicciones de los productos que están en ambos meses - Sólo productos añadidos
    both_prods = pred_prods - prev_prods
    both_prods = (both_prods > 0) * 1
    
    if env == 'submit':
        
        preds[index_last] = both_prods
    
        pred_probs = (preds * probs).argsort(axis=1)
        pred_probs = np.fliplr(pred_probs)[:, :7]
    
        targets = np.array(df_targets.columns.tolist())
        final_pred = [" ".join(list(targets[p])) for p in pred_probs]

        df_subm = pd.DataFrame({'ncodpers': df_test.ncodpers.values, 'added_products': final_pred})
        name_file = path + time.strftime("%Y-%m-%d-h%H-%M-%S_") + "submission.csv"
        df_subm.to_csv(name_file, index=False)
    
        return df_subm
    else:
        y_test.reset_index(drop=True, inplace=True)
        y = y_test.loc[index_last].as_matrix()
        purchases = y - prev_prods
        purchases = (purchases > 0) * 1        
        
        indexs = np.array([[i for i in range(y.shape[1])] * y.shape[0]]).reshape(y.shape)
        actual = purchases * indexs
        actual = list(map(lambda x: list(np.unique(x)[1:]), actual))
        
        preds = both_prods #Estas son las predicciones de los productos que se añaden 
        probs = probs[index_last]
        
        pred_probs = (preds * probs).argsort(axis=1)
        pred_probs = np.fliplr(pred_probs)[:, :7]        
        
        predicted = pred_probs
        
        return predicted, actual
"""
From github: @benhamner
"""

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])