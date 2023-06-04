import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
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

# Создание и обучение модели AdaBoost
model = AdaBoostClassifier()
model.fit(X_train, y_train)

# Прогнозирование классов на тестовом наборе данных
y_pred = model.predict(X_test)

# Вычисление и вывод метрик
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
print("Accuracy:", accuracy)
print("Precision:", precision)


# Подбор оптимальных параметров с помощью GridSearchCV
param_grid = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.1, 0.5, 1.0]
}

grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Лучшие параметры модели
best_params = grid_search.best_params_
print("Лучшие параметры:", best_params)

# Прогнозирование классов на тестовом наборе данных с использованием модели с лучшими параметрами
y_pred_best = grid_search.predict(X_test)

# Вычисление и вывод метрик для модели с лучшими параметрами
accuracy_best = accuracy_score(y_test, y_pred_best)
precision_best = precision_score(y_test, y_pred_best, average='macro')
print("Accuracy (best):", accuracy_best)
print("Precision (best):", precision_best)
