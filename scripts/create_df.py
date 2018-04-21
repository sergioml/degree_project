import numpy as np
import pandas as pd
from time import time

df = pd.read_csv('../data/clean/train_clean_v2.csv')
print('Data train cargado')
df_copy = df.copy()
df_target = pd.read_csv('../data/clean/train_labels.csv')
print("Targets cargados")

ncodpers = df_copy.ncodpers.value_counts()

first_records = {}
for i in range(17):
    first_records[i+1] = ncodpers.index[np.unique(ncodpers.values, return_index=True)[1][i]]

records_per_amount = {}
for i in np.unique(ncodpers.values, return_index=True)[0]:
    if i == 1:
        records_per_amount[i] = ncodpers.index[np.where(ncodpers.index == first_records[i])[0][0]:]
    else:
        records_per_amount[i] = ncodpers.index[np.where(ncodpers.index == first_records[i])[0][0]:np.where(ncodpers.index == first_records[i-1])[0][0]:]
        
only_record = df_copy.loc[df_copy['ncodpers'].isin(records_per_amount[1])].sort_values(['ncodpers', 'fecha_dato']).index
df_copy.drop(only_record, inplace=True)

group_ncodpers = df_copy.groupby('ncodpers')
print(len(group_ncodpers.groups.keys()), 'ncodpers')

start = time()
df_new = pd.DataFrame()

for i in range(2, len(group_ncodpers.groups.keys())+1):
    start2 = time()
    for ncod in records_per_amount[i]:
        df_aux = group_ncodpers.get_group(ncod) #DF con los registros de un código
        df_aux1 = df_aux.iloc[:-1] #Se obtienen la info de los clientes menos del último mes
        
        df_aux_target = df_target.iloc[df_aux.index[1:]] #Se obtienen los productos desde el segundo mes en adelante
        df_aux_target.set_index(df_aux1.index, inplace=True) #Se actualizan los índices del anterior df con los del aux1
        
        df_new = df_new.append(pd.concat([df_aux1, df_aux_target], axis=1))
    end2 = time()
    partial_time = end2 - start2

    print("Con", i, "se demoró", partial_time, "segundos", (partial_time)/float(60), "minutos")
    print("-----------------------------------------------------------------------------------")  
    
end = time()
print("Tiempo total", (end-start)/3600, "hrs")
#df_new.to_csv("../data/clean/new_df2.csv", index=False)
#print("Nuevo dataset creado")
