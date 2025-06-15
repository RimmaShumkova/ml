import numpy as np
import pandas as pd

#загрузка данных
data = pd.read_csv('SalaryC.csv')

data.info()
data.head()

print(data.isnull().sum()) #проверка пустых записей

#кодирование категориальных признаков
from sklearn.preprocessing import LabelEncoder, StandardScaler

label_encoder = LabelEncoder()
data['workclass'] = label_encoder.fit_transform(data['workclass'])
data['education'] = label_encoder.fit_transform(data['education'])
data['marital-status'] = label_encoder.fit_transform(data['marital-status'])
data['occupation'] = label_encoder.fit_transform(data['occupation'])
data['relationship'] = label_encoder.fit_transform(data['relationship'])
data['race'] = label_encoder.fit_transform(data['race'])
data['sex'] = label_encoder.fit_transform(data['sex'])
data['native-country'] = label_encoder.fit_transform(data['native-country'])
data['salary'] = label_encoder.fit_transform(data['salary'])

data.describe()

data.head()

#масштабирование числовых признаков
num_features = ['fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
scaler = StandardScaler()
data[num_features] = scaler.fit_transform(data[num_features])

print(data.describe())

import matplotlib.pyplot as plt
import seaborn as sns

data[num_features].hist(bins=20, figsize=(10, 8)) #гистограммы для числовых признаков
plt.tight_layout()
plt.show()

data_corr = data.drop(['fnlwgt', 'race', 'native-country'], axis=1)

corr_matrix = data_corr.corr() 
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

sns.countplot(x='salary', data=data) #анализ целевой переменной
plt.show()
print(data['salary'].value_counts())

#отбор признаков и масштабирование
X = data.drop(['salary'], axis=1)
y = data['salary']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.decomposition import PCA

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_
culminative_variance = explained_variance.cumsum()

n_components = (culminative_variance <= 0.95).sum() + 1
print(f'Число выбранных компонент: {n_components}')

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42) #разделение данных

#оценка базовой модели
base_model = LogisticRegression()
base_model.fit(X_train, y_train)
y_pred = base_model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1: {f1_score(y_test, y_pred):.4f}")

params_lr_grid = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}
lr = LogisticRegression(solver='liblinear')
grid_lr = GridSearchCV(lr, params_lr_grid, cv=5)
grid_lr.fit(X_train, y_train)

params_rf_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
rf = RandomForestClassifier()
grid_rf = GridSearchCV(rf, params_rf_grid, cv=5)
grid_rf.fit(X_train, y_train)

params_xgb_grid = {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.01]}
xgb = XGBClassifier()
grid_xgb = GridSearchCV(xgb, params_xgb_grid, cv=5)
grid_xgb.fit(X_train, y_train)

models = {'Logistic Regression': grid_lr.best_estimator_, 'Random Forest': grid_rf.best_estimator_, 'XGBoost': grid_xgb.best_estimator_}

for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f'{name}:')
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1: {f1_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}\n")
    
import joblib

joblib.dump(grid_xgb.best_estimator_, 'best_xgb_model.pkl')