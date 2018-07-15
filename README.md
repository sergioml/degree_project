# Degree project
Degree project in Systems Engineering.

**Title:** Banking products recommendation system with Machine Learning techniques  
**Autor:** Sergio Martínez Lizarazo  
**Degree:** Systems Engineering

---

### Notebooks de jupyter
Descripción de los notebooks de jupyter

* `data_exploration:` Primera exploración de los datos
* `data_exploration_V2:` Limpieza de los datos, eliminación de columnas y filas de datos inconsistentes y generación de nuevos datasets
* `first_attempts:` Primeros entrenamientos con un _RandomForestClassifier_
* `keep_working_1:` Generación de un modelo por cada producto que hay, se sigue usando el algoritmo Random Forest
* `keep_working_2:` Entrenamiento de los datos con targets balanceados
* `keep_working_3:` Entrenamiento y test, con métricas de sensibilidad  y especifidad
* `keep_working_4:` Procesamiento de dataset de entrenamiento añadiendo los productos del mes anterior como *features*
* `keep_working_5:` Exploración y análisis más avanzado de datos.
* `test_data_preprocessed:` Procesamiento del dataset de test suministrado por Kaggle y tiene una versión <span style='color:red'>**obsoleta**</span> de la generación del archivo de submission
* `local_environment:` Notebook con un ambiente local de pruebas, incluida la métrica usada en Kaggle
* `dont_give_up_1:` Experimentos iniciales de la segunda etapa
* `dont_give_up_2:` Experimentos de la segunda etapa en la que se está usando el `local_environment.py`
---
### Scripts
Descripción de los scripts de python utilizados

* `python.batch:` Script en batch para lanzar jobs de python
* `df_test.py:` Preprocesado del dataset de test suministrado por Kaggle
* `metrics.py:` Métricas de test como true negative, false positive, false negative y true positive
* `submission.py:` <span style='color:green'>**Pendiente por rediseñar por ahora OBSOLETO**</span> Script para hacer submissiones en Kaggle
* `targets_balanced.py:` Prueba con un RandomForestClassifier y los targets balanceados
* `subm_pca.py:` Prueba con PCA
* `subm_prev_prods.py:` Prueba con los productos del mes anterior como features
* `local_environment.py:` Script con las funciones necesarias para el entrenamiento, testeo y generación de archivos de submission
* `experiment6.py:` Experimento progresivo individual, en ambiente local
* `experiment7.py:` Experimento progresivo acumulado, en ambiente local
