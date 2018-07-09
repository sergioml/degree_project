import numpy as np
import pandas as pd
from submission import submission

from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('../data/clean/train_clean.csv')
df_target = pd.read_csv('../data/clean/train_labels.csv')
df_test = pd.read_csv('../data/clean/test_clean.csv')

x_train = df.drop(['fecha_dato', 'ncodpers', 'fecha_alta'], axis=1)
x_test = df_test.drop(['fecha_dato', 'fecha_alta'], axis=1)

ncodpers_last_month = df[df['fecha_dato'] == df['fecha_dato'].unique()[-1]]['ncodpers']

rf = RandomForestClassifier(n_jobs=4)

submission(x_train=x_train,
           y_train=df_target,
           x_test=x_test,
           clf=rf,
           ncodpers_last_month=ncodpers_last_month)
