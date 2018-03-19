import numpy as np
import pandas as pd
from time import time

df = pd.read_csv('../data/clean/train_clean_v2.csv')
print('Data train cargado')
df_copy = df.copy()
df_target = pd.read_csv('../data/clean/train_labels.csv')
print("Targets cargado")

all_records = df_copy.ncodpers.value_counts().index

first_records = {} #Primer registro de cada cantidad
for i in range(17):
    first_records[i+1] = df_copy.ncodpers.value_counts().index[np.unique(df_copy.ncodpers.value_counts().values, return_index=True)[1][i]]

l_records = {} #los indices según la cantidad que hay de cada uno
for i in np.unique(df_copy.ncodpers.value_counts().values, return_index=True)[0]:
    if i == 1:
        l_records[i] = all_records[np.where(all_records==first_records[1])[0][0]:]
    else:
        l_records[i] = all_records[np.where(all_records==first_records[i])[0][0]:np.where(all_records==first_records[i-1])[0][0]]

only_record = df_copy.loc[df_copy['ncodpers'].isin(l_records[1])].sort_values(['ncodpers', 'fecha_dato']).index
df_copy.drop(only_record, inplace=True)

group_ncodpers = df_copy.groupby('ncodpers')
group_fecha = df_copy.groupby('fecha_dato')

df_new = pd.DataFrame()

print("Nuevo dataset creado")

for i in range(2, len(l_records)+1):
    start2 = time()
    lista = l_records[i]
    for code in lista:
        df_aux = group_ncodpers.get_group(code) #DF con los registros de un código

        df_aux1 = df_aux.iloc[:-1] #Se obtienen los registros menos el ltimo

        df_aux_target = df_target.iloc[df_aux.index].iloc[1:] #Se obtienen los targets desde el segundo en adelante
        df_aux_target.set_index(df_aux1.index, inplace=True) #Se actualizan los índices del anterior df con los del aux1

        df_aux3 = pd.concat([df_aux1, df_aux_target], axis=1) #Se juntan ambos df resultantes

        df_new = df_new.append(df_aux3) #Se concatenan entre sí
    end2 = time()
    partial_time = end2 - start2

    print("Con", i, "se demoró", partial_time, "segundos", (partial_time)/float(60), "minutos")

df_new.to_csv("../data/clean/new_df.csv", index=False)
