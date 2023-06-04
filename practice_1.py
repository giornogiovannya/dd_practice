import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv('Data_Cortex_Nuclear.csv')
print(data)

# 1
sns.countplot(x='class', data=data)
plt.show()

sns.countplot(x='Treatment', data=data)
plt.show()
sns.countplot(x='Behavior', data=data)
plt.show()

# 2
sns.histplot(data=data, x='DYRK1A_N')
plt.show()
sns.histplot(data=data, x="BDNF_N")
plt.show()

# 3
numeric_features = data.iloc[:, 1:6]
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(numeric_features)
print(pd.DataFrame(normalized_data, columns=numeric_features.columns).head())

# 4
corr = data.corr(numeric_only=True)
sns.heatmap(corr, cmap="crest")
plt.show()
