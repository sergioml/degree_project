import numpy as np
import pandas as pd

df_test = pd.read_csv('../data/test_ver2.csv', dtype={'sexo': str, 'age': str, 'ind_nuevo': str,
                                                    'indrel_1mes': str, 'antiguedad': str, 'ult_fec_cli_lt': str,
                                                    'indext': str, 'conyuemp': str}, parse_dates=['fecha_dato', 'fecha_alta'])

#Eliminar columnas
df_test.drop(labels=['conyuemp', 'ult_fec_cli_1t'], inplace=True, axis=1)

#Eliminar filas
missing = df_test[df_test.isnull().any(axis=1)]
missing_index = missing.index

df_test.drop(missing_index, inplace=True, axis=0)

#Cambio de tipo de variable
df_test.age = df_test.age.astype('int64')
df_test.ind_nuevo = df_test.ind_nuevo.astype('int64')
df_test.antiguedad = df_test.antiguedad.astype('int64')
df_test.indrel_1mes = df_test.indrel_1mes.astype('float64')

#Cambio de atributos categóricos a numéricos
cols = df_test.select_dtypes(['object']).columns

for col in cols:
    print(col, '¡LISTO!')
    attribute_vals = df_test[col].value_counts().index
    attribute_counts = np.arange(1, len(attribute_vals) + 1)
    df_test.replace(attribute_vals, attribute_counts, inplace=True)
    
#Cambio de formato de fecha
df_test['fecha_dato_year'] = df_test.fecha_dato.dt.year
df_test['fecha_dato_month'] = df_test.fecha_dato.dt.month
df_test['fecha_dato_day'] = df_test.fecha_dato.dt.day

df_test['fecha_alta_year'] = df_test.fecha_alta.dt.year
df_test['fecha_alta_month'] = df_test.fecha_alta.dt.month
df_test['fecha_alta_day'] = df_test.fecha_alta.dt.day

#Guardar
df_test.to_csv('../data/clean/test_clean.csv', index=False)

