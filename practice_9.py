import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score
from sklearn.impute import SimpleImputer

data = pd.read_csv("Data_Cortex_Nuclear.csv")
imputer = SimpleImputer(strategy='mean')

# Выделение признаков и целевой переменной
features = data.iloc[:, 1:78]
target = data["class"]

# Разбиение на train/test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Заполнение пропущенных значений
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Создание и обучение моделей
random_forest = RandomForestClassifier()
qda = QuadraticDiscriminantAnalysis()

# Объединение моделей в ансамбль
ensemble = VotingClassifier(estimators=[('rf', random_forest), ('qda', qda)], voting='hard')
ensemble.fit(X_train, y_train)

# Прогнозирование классов на тестовом наборе данных
y_pred = ensemble.predict(X_test)

# Вычисление и вывод метрик
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
print("Accuracy:", accuracy)
print("Precision:", precision)

# Подбор оптимальных параметров для случайного леса с помощью RandomizedSearchCV
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

randomized_search = RandomizedSearchCV(random_forest, param_grid, cv=5, n_iter=10)
randomized_search.fit(X_train, y_train)

# Лучшие параметры для случайного леса
best_params = randomized_search.best_params_
print("Лучшие параметры для случайного леса:", best_params)

# Прогнозирование классов на тестовом наборе данных с использованием модели случайного леса с лучшими параметрами
y_pred_best = randomized_search.predict(X_test)

# Вычисление и вывод метрик для модели случайного леса с лучшими параметрами
accuracy_best = accuracy_score(y_test, y_pred_best)
precision_best = precision_score(y_test, y_pred_best, average='macro')
print("Accuracy (best) для случайного леса:", accuracy_best)
print("Precision (best) для случайного леса:", precision_best)

# Подбор оптимальных параметров для квадратичного дискриминантного анализа с помощью RandomizedSearchCV
param_grid_qda = {
    "reg_param": [0.0, 0.1, 0.5, 1.0]
}

randomized_search_qda = RandomizedSearchCV(qda, param_grid_qda, cv=5, n_iter=10)
randomized_search_qda.fit(X_train, y_train)

# Лучшие параметры для квадратичного дискриминантного анализа
best_params_qda = randomized_search_qda.best_params_
print("Лучшие параметры для квадратичного дискриминантного анализа:", best_params_qda)

# Прогнозирование классов на тестовом наборе данных с использованием модели квадратичного дискриминантного анализа с лучшими параметрами
y_pred_best_qda = randomized_search_qda.predict(X_test)

# Вычисление и вывод метрик для модели квадратичного дискриминантного анализа с лучшими параметрами
accuracy_best_qda = accuracy_score(y_test, y_pred_best_qda)
precision_best_qda = precision_score(y_test, y_pred_best_qda, average='macro')
print("Accuracy (best) для квадратичного дискриминантного анализа:", accuracy_best_qda)
print("Precision (best) для квадратичного дискриминантного анализа:", precision_best_qda)
