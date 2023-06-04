import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
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

# Создание и обучение модели нейронной сети
model = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42)

# Обучение модели и запись ошибок на каждой эпохе
errors = []
for epoch in range(1, 1001):
    model.partial_fit(X_train, y_train, classes=np.unique(y_train))
    y_pred_train = model.predict(X_train)
    error = 1 - accuracy_score(y_train, y_pred_train)
    errors.append(error)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Error: {error}")

# Прогнозирование классов на тестовом наборе данных
y_pred = model.predict(X_test)

# Вычисление и вывод метрик
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
print("Accuracy:", accuracy)
print("Precision:", precision)

# Построение графика изменения ошибки с ростом количества эпох
plt.plot(range(1, 1001), errors)
plt.xlabel("Эпоха")
plt.ylabel("Ошибка")
plt.title("Эпох vs. Ошибок")
plt.show()
