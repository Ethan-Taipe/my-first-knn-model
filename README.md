# Optimización y Evaluación de un Modelo KNN

Este proyecto consiste en la construcción, optimización y evaluación de un modelo de clasificación utilizando el algoritmo **K-Nearest Neighbors (KNN)** sobre un dataset personalizado.

## 📁 Archivo principal

- `Modelo_knn_optimizado.ipynb`: Notebook donde se desarrolla todo el proceso de análisis, optimización de hiperparámetros y evaluación final del modelo.

---

## 🧠 Objetivos del proyecto

- Entrenar un modelo KNN con un valor fijo de `n_neighbors = 11`.
- Explorar combinaciones óptimas de hiperparámetros mediante:
  - `GridSearchCV`
  - `RandomizedSearchCV`
- Evaluar el desempeño del modelo utilizando validación cruzada.
- Visualizar las 5 mejores combinaciones de hiperparámetros.
- Reentrenar el modelo con la mejor combinación encontrada.
- Evaluar el rendimiento en el conjunto de prueba (accuracy y matriz de confusión).
- Guardar el modelo final entrenado utilizando `joblib`.

---

## 🔧 Hiperparámetros explorados

Se optimizaron los siguientes hiperparámetros del modelo `KNeighborsClassifier`:

- `weights`: ['uniform', 'distance']
- `algorithm`: ['auto', 'ball_tree', 'kd_tree', 'brute']
- `metric`: ['euclidean', 'manhattan', 'chebyshev', 'minkowski']

---

## 🧪 Evaluación del modelo

- Se utilizó **validación cruzada con 6 folds**.
- La métrica de evaluación fue la **precisión (accuracy)**.
- Se compararon múltiples combinaciones de hiperparámetros.
- Se graficaron los 5 mejores resultados tanto para GridSearch como para RandomizedSearch.

---

## ✅ Resultados

- Mejores hiperparámetros encontrados:
  ```python
  {'algorithm': 'brute', 'metric': 'manhattan', 'weights': 'distance'}
