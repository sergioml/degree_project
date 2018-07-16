import numpy as np
import pandas as pd

df = pd.read_csv('../data/train_ver2.csv',
                 dtype={'sexo': str, 'age': str, 'ind_nuevo': str, 'indrel_1mes': str, 
                        'antiguedad': str, 'ult_fec_cli_lt': str, 'indext': str, 'conyuemp': str},
                 parse_dates=['fecha_dato', 'fecha_alta'])

df.drop(labels=['conyuemp', 'ult_fec_cli_1t'], inplace=True, axis=1)

renta_by_seg = df.groupby(['nomprov', 'segmento']).agg({'renta' : 'mean'})
renta_by_seg = renta_by_seg.loc[:, ['renta']].round(2)

for i in renta_by_seg.index:
    renta_value = renta_by_seg.loc[i, 'renta']
    df.loc[(df['nomprov'] == i[0]) & (df['segmento'] == i[1]) & (df['renta'].isnull()), 'renta'] = renta_value

missing = df[df.isnull().any(axis=1)]
df.drop(missing.index, inplace=True, axis=0)

df[['ind_nomina_ult1', 'ind_nom_pens_ult1']] = df[['ind_nomina_ult1', 'ind_nom_pens_ult1']].astype('int64')
df['age'] = df['age'].astype('int64')
df['ind_nuevo'] = df['ind_nuevo'].astype('int64')
df['antiguedad'] = df['antiguedad'].astype('int64')
df['indrel_1mes'] = df['indrel_1mes'].astype('float64')

df['fecha_dato_year'] = df.fecha_dato.dt.year
df['fecha_dato_month'] = df.fecha_dato.dt.month
df['fecha_dato_day'] = df.fecha_dato.dt.day

df['fecha_alta_year'] = df.fecha_alta.dt.year
df['fecha_alta_month'] = df.fecha_alta.dt.month
df['fecha_alta_day'] = df.fecha_alta.dt.day

df_cols = df.columns.tolist()
df_cols_reordered = df_cols[:1] + df_cols[46:49] + df_cols[1:7] + df_cols[49:52] + df_cols[7:46]
df = df[df_cols_reordered]

df = df.loc[:, :'segmento']
cols_object = df.select_dtypes(['object']).columns

for col in cols_object:
    attribute_vals = df[col].value_counts().index
    attribute_counts = np.arange(1, len(attribute_vals) + 1)
    df.replace(attribute_vals, attribute_counts, inplace=True)
    print(col, '¡LISTO!')

df.to_csv('../data/clean/train_clean_v2.csv', index=False)
