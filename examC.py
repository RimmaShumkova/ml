import numpy as np
import pandas as pd

#загрузка данных
data = pd.read_csv('SalaryC.csv')

data.info()
data.head()

print(data.isnull().sum()) #проверка пустых записей

duplicates = data.duplicated()
print(f"Найдено {duplicates.sum()} дубликатов")

data = data.drop_duplicates()
duplicates = data.duplicated()
print(f"Количество дубликатов после удаления: {duplicates.sum()}")

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

import matplotlib.pyplot as plt
import seaborn as sns

#визуализация признаков
num_features = ['age', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'hours-per-week']
data[num_features].hist(bins=20, figsize=(10,8))
plt.show()

#избавляемся от выбросов
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

data[num_features].hist(bins=20, figsize=(10,8))
plt.show()

print(data.describe())

#матрица корреляции
data_corr = data.drop(['fnlwgt', 'race', 'native-country'], axis=1)

corr_matrix = data_corr.corr() 
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

sns.countplot(x='salary', data=data) #анализ целевой переменной
plt.show()
print(data['salary'].value_counts())

#реализация regplot
target_column = 'salary'

# Расчет корреляции и выбор топ-3 признаков
corr_matrix = data.corr()
target_corr = corr_matrix[target_column].drop(target_column)
top_3_features = target_corr.abs().nlargest(3).index

# Построение графиков с линией регрессии
for feature in top_3_features:
    plt.figure(figsize=(8, 6))
    sns.regplot(  # Используем regplot вместо scatterplot для линии регрессии
        x=data[feature], 
        y=data[target_column],
        scatter_kws={'alpha': 0.5},  # Прозрачность точек
        line_kws={'color': 'red'}    # Цвет линии регрессии
    )
    plt.title(f'Зависимость: {feature} vs {target_column}\n(Корреляция: {target_corr[feature]:.2f})')
    plt.xlabel(feature)
    plt.ylabel(target_column)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

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
from imblearn.over_sampling import SMOTE


X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42) #разделение данных

smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)

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

loded_model = joblib.load('best_xgb_model.pkl')
sample = X_pca[0:1]
print(f'Предсказание: {loded_model.predict(sample)}')

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
Почему я выбрала метод понижения размерности PCA:

Эффективность с количественными признаками: PCA особенно хорошо работает с числовыми данными, которые у нас были после предварительной обработки (масштабирования через StandardScaler).

Сохранение дисперсии: Я проанализировала кумулятивную дисперсию и выбрала количество компонент, которое сохраняет 95% исходной дисперсии данных (у нас получилось n_components компонент). Это позволило значительно уменьшить размерность без существенной потери информации.

Корреляция признаков: На тепловой карте корреляций было видно, что некоторые признаки коррелируют между собой. PCA помогает решить проблему мультиколлинеарности, преобразуя признаки в ортогональные компоненты.

Улучшение производительности моделей: Уменьшение размерности часто приводит к ускорению обучения моделей без потери качества предсказаний.

Почему я выбрала именно эти три модели:

Logistic Regression:

Простая и интерпретируемая модель, хорошая базовая точка для сравнения

Хорошо работает с линейно разделимыми данными

Эффективна после применения PCA, так как преобразованные признаки часто лучше соответствуют линейным предположениям

Random Forest:

Устойчив к переобучению и хорошо работает с данными, где есть сложные нелинейные зависимости

Не требует тщательной настройки гиперпараметров для достижения хорошего результата

Может работать с важностью признаков, что полезно после PCA

Автоматически обрабатывает взаимодействия между признаками

XGBoost:

Одна из самых мощных алгоритмов для задач классификации

Имеет встроенные механизмы борьбы с переобучением

Хорошо работает с несбалансированными данными (что было важно в нашем случае, как показал анализ целевой переменной)

Позволяет точно настраивать различные аспекты модели через множество гиперпараметров

Дополнительные обоснования:

Для борьбы с дисбалансом классов я использовала SMOTE, что особенно важно для метрик F1 и Precision

Для всех моделей проводился подбор гиперпараметров через GridSearchCV для достижения наилучшего результата

Итоговые метрики (Accuracy, F1, Precision) показали, что выбранный подход эффективен для данной задачи
'''

'''
ссылки на билеты:
1. https://github.com/RimmaShumkova/ml/blob/main/1.txt (Искусственный интеллект (Artificial Intelligence, AI). Большие данные, знания. Технологии AI. Типы AI.)
2. https://github.com/RimmaShumkova/ml/blob/main/2.txt (Машинное обучение (Machine Learning, ML). Типы ML. Типы задач в ML. Примеры задач)
3. https://github.com/RimmaShumkova/ml/blob/main/3.txt (Основные понятия ML. Обучающая, валидационная и тестовая выборки. Кросс-валидация. Сравнительная характеристика методов k-fold и holdout.)
4. https://github.com/RimmaShumkova/ml/blob/main/4.txt (Обучение с учителем. Проблемы ML. Решение проблемы переобучения.)
5. https://github.com/RimmaShumkova/ml/blob/main/5.txt (Разведочный анализ данных (EDA). Понятие EDA. Цель и этапы EDA. Основы описательной статистики в EDA.)
6. https://github.com/RimmaShumkova/ml/blob/main/6.txt (Типы данных в EDA. Предобработка данных. Инструменты визуализации EDA. Визуализация зависимостей признаков в наборе данных.)
7. https://github.com/RimmaShumkova/ml/blob/main/7.txt (Задача регрессии. Линейная модель. Метод наименьших квадратов. Функция потерь. Метрики оценки регрессии.)
8. https://github.com/RimmaShumkova/ml/blob/main/8.txt (Задача регрессии. Многомерная линейная регрессия, проблема мультиколлинеарности. Регуляризованная регрессия. Полиномиальная регрессия.)
9. https://github.com/RimmaShumkova/ml/blob/main/9.txt (Задача классификации. Алгоритмы классификации в ML. Проблема дисбаланса классов и ее решение. Методы сэмплирования. Метрики оценки классификации. Ошибки классификации.)
10. https://github.com/AvdushkinaKsenia/ml/blob/main/10.txt (Логистическая регрессия — это алгоритм классификации, который предсказывает вероятность принадлежности объекта к определённому классу.)
11. https://github.com/AvdushkinaKsenia/ml/blob/main/11.txt (Метрический классификатор (англ. similarity-based classifier) — алгоритм классификации, основанный на вычислении оценок сходства между объектами.)
12. https://github.com/AvdushkinaKsenia/ml/blob/main/12.txt (SVM)
13. https://github.com/AvdushkinaKsenia/ml/blob/main/13.txt (Decision Tree)
14. https://github.com/AvdushkinaKsenia/ml/blob/main/14.txt (Ансамблевое обучение)
15. https://github.com/AvdushkinaKsenia/ml/blob/main/15.txt (Бустинг, градиентный бустинг, AdaBoost и т.д)
16. https://github.com/AvdushkinaKsenia/ml/blob/main/16.txt (Обучение без учителя, кластеризация)
17. https://github.com/AvdushkinaKsenia/ml/blob/main/17.txt (Кластеризация, алгоритмы, k-means, иерархическая)

'''