import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = pd.read_csv('MaintenanceR.csv') #загрузка файла
print(data.isnull().sum()) #проверка нулевых значений

data.head()

duplicates = data.duplicated()
print(f"Найдено {duplicates.sum()} дубликатов")

data = data.drop_duplicates()
print(f"Количество дубликатов после удаления: {duplicates.sum()}")

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

### ДЛЯ СПРАВКИ. ###

#ЕСЛИ ЕСТЬ ПУСТЫЕ СТРОКИ

data_cleaned = data.dropna() # Удаление всех строк, где хотя бы один столбец содержит NaN
data_cleaned = data.dropna(how='all') # Удаление строк, где все значения NaN

#АЛЬТЕРНАТИВА УДАЛЕНИЮ - ЗАПОЛНЕНИЕ ЗНАЧЕНИЙ

data.fillna(data.mean(), inplace=True) # Заполнение средним/медианным
data.interpolate(method='linear', inplace=True) # Или интерполяция

#ЕСЛИ НУЖНО ПОМЕНЯТЬ ТИП ДАННОГО

data['free sulfur dioxide'] = data['free sulfur dioxide'].astype(int) #float в int
data["Type"] = data["Type"].map({"L": 0, "M": 1, "H": 2}) #object в int

''' 
Почему был выбран метод понижения размерности PCA:

Обработка мультиколлинеарности: На тепловой карте корреляций было видно наличие взаимосвязей между некоторыми признаками (например, между температурой воздуха и крутящим моментом). PCA помогает устранить эту проблему, создавая ортогональные компоненты.

Оптимальное сохранение информации: Анализ кумулятивной дисперсии показал, что выбранное количество компонент (n_comp) сохраняет 95% исходной информации, что является хорошим компромиссом между сокращением размерности и сохранением полезной вариативности данных.

Ускорение обучения моделей: Уменьшение количества признаков позволило ускорить процесс обучения, особенно для таких "тяжелых" алгоритмов, как Random Forest и XGBoost, без существенной потери качества предсказаний.

Улучшение обобщающей способности: Понижение размерности помогает бороться с "проклятием размерности" и улучшает устойчивость моделей к переобучению.

Обоснование выбора моделей регрессии:

ElasticNet:

Был выбран как базовая линейная модель, сочетающая L1 и L2 регуляризацию

Хорошо работает после PCA, так как преобразованные признаки часто лучше соответствуют предположениям линейных моделей

Устойчив к мультиколлинеарности (что особенно важно для нашей задачи)

Позволяет оценить вклад каждого признака через коэффициенты

Random Forest Regressor:

Способен улавливать сложные нелинейные зависимости в данных

Устойчив к выбросам (которые могли остаться после предобработки)

Не требует тщательной предобработки данных и нормализации

Автоматически определяет важность признаков

Дает хорошие результаты "из коробки" с минимальной настройкой

XGBRegressor:

Один из самых мощных алгоритмов для задач регрессии

Имеет встроенные механизмы регуляризации для борьбы с переобучением

Эффективно работает с различными масштабами признаков

Поддерживает раннюю остановку для оптимизации времени обучения

Показывает отличные результаты на структурированных данных

Дополнительные обоснования выбранного подхода:

Для всех моделей проводился тщательный подбор гиперпараметров через GridSearchCV

Использовались адекватные метрики для задачи регрессии: MSE, MAE и R²

Лучшая модель (Random Forest) была сохранена для последующего использования
'''

'''
ссылки на билеты:
1. https://github.com/RimmaShumkova/ml/blob/main/1.txt (Искусственный интеллект (Artificial Intelligence, AI). Большие данные, знания. Технологии AI. Типы AI.)
2. https://github.com/RimmaShumkova/ml/blob/main/2.txt (Машинное обучение (Machine Learning, ML). Типы ML. Типы задач в ML. Примеры задач. Жизненный цикл модели ML. Схема проекта по ML.)
3. https://github.com/RimmaShumkova/ml/blob/main/3.txt (Основные понятия ML. Обучающая, валидационная и тестовая выборки. Кросс-валидация. Сравнительная характеристика методов k-fold и holdout.)
4. https://github.com/RimmaShumkova/ml/blob/main/4.txt (Обучение с учителем. Проблемы ML. Решение проблемы переобучения.)
5. https://github.com/RimmaShumkova/ml/blob/main/5.txt (Разведочный анализ данных (EDA). Понятие EDA. Цель и этапы EDA. Основы описательной статистики в EDA.)
6. https://github.com/RimmaShumkova/ml/blob/main/6.txt (Типы данных в EDA. Предобработка данных. Инструменты визуализации EDA. Визуализация зависимостей признаков в наборе данных.)
7. https://github.com/RimmaShumkova/ml/blob/main/7.txt (Задача регрессии. Линейная модель. Метод наименьших квадратов. Функция потерь. Метрики оценки регрессии.)
8. https://github.com/RimmaShumkova/ml/blob/main/8.txt (Задача регрессии. Многомерная линейная регрессия, проблема мультиколлинеарности. Регуляризованная регрессия. Полиномиальная регрессия.)
9. https://github.com/RimmaShumkova/ml/blob/main/9.txt (Задача классификации. Алгоритмы классификации в ML. Проблема дисбаланса классов и ее решение. Методы сэмплирования. Метрики оценки классификации. Ошибки классификации.)
'''