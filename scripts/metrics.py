import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('../data/clean/train_clean_v2.csv')
df_targets = pd.read_csv('../data/clean/train_labels.csv')

df_copy = df.copy()

print("Datasets' shape")
print(df.shape, df_targets.shape)
print()

x = df_copy.drop(['fecha_dato', 'fecha_alta', 'ncodpers'], axis=1).as_matrix()
y = df_targets.as_matrix()

def metrics(x_train, y_train, x_test, y_test, clf):

    #INPUTS
    #x_train: data to train
    #y_train: target, one column
    #x_test: data to test
    #y_test: target to test, one column
    #clf: classifier

    clf.fit(x_train, y_train)
    preds = [clf.predict(row.reshape(1, -1))[0] for row in x_test]

    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    sensitivity = tp / (tp + fn)
    specifity = tn / (tn + fp)

    #OUTPUTS
    #preds (list): predictions for each row
    #tn: true negative
    #fp: false positive
    #fn: false negative
    #tp: true positive
    return tn, fp, fn, tp, sensitivity, specifity

x_train = x[:10094948]
x_test = x[10094948:]
y_train = y[:10094948]
y_test = y[10094948:]

print("Negative is 0 and Positive is 1")

for index, row in enumerate(df_targets.columns):
    rf = RandomForestClassifier()
    tn, fp, fn, tp, sensitivity, specifity = metrics(x_train, y_train[:, index], x_test, y_test[:, index], rf)
    print(row)
    print('True positive:', tp)
    print('False positive:', fp)
    print('True positive rate or sensitivity:', sensitivity)
    print('True negative:', tn)
    print('False negative:', fn)
    print('True negative rate or specifity:', specifity)
    print()
