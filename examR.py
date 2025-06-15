import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = pd.read_csv('MaintenanceR.csv') #загрузка файла
print(data.isnull().sum()) #проверка нулевых значений

data.head()

data = data.drop(['UDI', 'Product ID'], axis=1)

#кодирование категориальных признаков
label_encoder = LabelEncoder()
data['Type'] = label_encoder.fit_transform(data['Type'])
data.head()

import seaborn as sns
import matplotlib.pyplot as plt

#анализ категориального признака Type
sns.countplot(x='Type', data=data)
plt.show()

#визуализация признаков
num_features = ['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
data[num_features].hist(bins=20, figsize=(10, 8))
plt.show()

#удаляем выбросы
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

data = remove_outliers(data, num_features)

#визуализация признаков
num_features = ['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
data[num_features].hist(bins=20, figsize=(10, 8))
plt.show()

scaler = StandardScaler() #стандартизация числовых признаков
data[num_features] = scaler.fit_transform(data[num_features])

print(data.describe()) #описание данных

data_corr = data.drop(['Type'], axis=1) #убираем ненужные столбцы

corr_matrix = data_corr.corr() #
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

sns.countplot(x='Process temperature [K]', data=data) #анализируем целевую переменную
plt.show()
print(data['Process temperature [K]'].value_counts())

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

#отбор признаков
X = data.drop(['Process temperature [K]'], axis=1)
y = data['Process temperature [K]']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.decomposition import PCA

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_
culminative_variance = explained_variance.cumsum()

n_comp = (culminative_variance <= 0.95).sum() + 1
print(f"Число выбранных компонент: {n_comp}")

pca = PCA(n_components=n_comp)
X_pca = pca.fit_transform(X_scaled)

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42) #разделение данных

#оценка параметров базовой модели
base_model = ElasticNet()
base_model.fit(X_train, y_train)
y_pred = base_model.predict(X_test)

print(f"R2: {r2_score(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")

#применение регрессоров и их гиперпараметрическая настройка
params_en_grid = {'alpha': [0.1, 1, 10], 'l1_ratio': [0.2, 0.5, 0.8]}
en = ElasticNet()
grid_en = GridSearchCV(en, params_en_grid, cv=5)
grid_en.fit(X_train, y_train)

params_rf_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
rf = RandomForestRegressor()
grid_rf = GridSearchCV(rf, params_rf_grid, cv=5)
grid_rf.fit(X_train, y_train)

params_xgb_grid = {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.01]}
xgb = XGBRegressor()
grid_xgb = GridSearchCV(xgb, params_xgb_grid, cv=5)
grid_xgb.fit(X_train, y_train)

#сравнение метрик
models = {'ElasticNet': grid_en.best_estimator_, 'RandomForestRegressor': grid_rf.best_estimator_, 'XGBRegressor': grid_xgb.best_estimator_}

for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f'{name}:')
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"R²: {r2_score(y_test, y_pred):.4f}\n")
    
#деплой модели
import joblib

joblib.dump(grid_rf.best_estimator_, 'best_model.pkl')

loaded_model = joblib.load('best_model.pkl')
sample = X_pca[0:1] #выбираем первую запись
print(f'Предсказание: {loaded_model.predict(sample)}')