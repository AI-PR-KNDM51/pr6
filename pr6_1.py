import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# Встановлення насіння для відтворюваності результатів
np.random.seed(42)

# Кількість зразків у наборі даних
n_samples = 299

# Створення словника з даними
data = {
    'вік': np.random.randint(40, 90, size=n_samples),
    'анемія': np.random.choice([0, 1], size=n_samples),
    'креатинін_фосфокіназа': np.random.randint(30, 800, size=n_samples),
    'діабет': np.random.choice([0, 1], size=n_samples),
    'фракція_викиду': np.random.randint(14, 80, size=n_samples),
    'високий_кров_тиск': np.random.choice([0, 1], size=n_samples),
    'тромбоцити': np.random.uniform(100, 400, size=n_samples).round(2),
    'сироватковий_креатинін': np.random.uniform(0.5, 2.5, size=n_samples).round(2),
    'сироватковий_натрій': np.random.randint(120, 150, size=n_samples),
    'стать': np.random.choice([0, 1], size=n_samples),
    'куріння': np.random.choice([0, 1], size=n_samples),
    'час': np.random.randint(30, 300, size=n_samples),
    'смертність': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])  # 0: Вижив, 1: Помер
}

# Створення DataFrame з даних
df = pd.DataFrame(data)

# Відділення цільової змінної від ознак
X = df.drop('смертність', axis=1)
y = df['смертність']

# Масштабування ознак для нормалізації
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Визначення розмірності вхідних даних та розміру кодування для автоенкодера
input_dim = X_scaled.shape[1]
encoding_dim = 5

# Створення архітектури автоенкодера
input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='linear')(encoded)

# Створення моделі автоенкодера
autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Навчання автоенкодера
autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=16, shuffle=True, verbose=0)

# Витягування енкодера з навченої моделі
encoder_model = Model(inputs=input_layer, outputs=encoded)
encoded_X = encoder_model.predict(X_scaled)

# Кластеризація з використанням K-Means
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(encoded_X)
df['Кластер'] = clusters

# Виведення описових статистик DataFrame
print(df.describe())

# Підрахунок кількості смертей та виживання
death_counts = df['смертність'].value_counts()
labels = ['Вижив', 'Помер']

# Побудова кругової діаграми розподілу смертності пацієнтів
plt.figure(figsize=(6,6))
plt.pie(death_counts, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Розподіл смертності пацієнтів')
plt.axis('equal')  # Забезпечення рівних осей для кола
plt.show()

# Обчислення кореляційної матриці
corr_matrix = df.corr()

# Візуалізація кореляційної матриці за допомогою теплової карти
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Кореляційна матриця')
plt.show()

# Вибір релевантних ознак з кореляцією більше 0.1 з цільовою змінною
corr_target = abs(corr_matrix['смертність'])
relevant_features = corr_target[corr_target > 0.1].index.tolist()
relevant_features.remove('смертність')  # Видалення цільової змінної зі списку ознак

# Відділення релевантних ознак та цільової змінної
X_reg = df[relevant_features]
y_reg = df['смертність']

# Переконвертування даних у числовий формат та видалення пропусків
X_reg = X_reg.apply(pd.to_numeric, errors='coerce')
X_reg = X_reg.dropna()
y_reg = y_reg.loc[X_reg.index]

# Розділення даних на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Створення та навчання моделі логістичної регресії
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Прогнозування на тестових даних
y_pred = log_reg.predict(X_test)

# Обчислення точності моделі
accuracy = accuracy_score(y_test, y_pred)
print(f'Точність моделі логістичної регресії: {accuracy:.2f}')

# Створення матриці сплутаності
cm = confusion_matrix(y_test, y_pred)

# Візуалізація матриці сплутаності за допомогою теплової карти
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Матриця Сплутаності')
plt.xlabel('Прогноз')
plt.ylabel('Реальні')
plt.show()

# Отримання ймовірностей прогнозу для ROC кривої
y_prob = log_reg.predict_proba(X_test)[:,1]

# Обчислення точок для ROC кривої
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Побудова ROC кривої
plt.figure()
plt.plot(fpr, tpr, label=f'ROC крива (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], 'k--')  # Діагональна лінія випадкової класифікації
plt.xlabel('Частота хибнопозитивних')
plt.ylabel('Чутливість')
plt.title('ROC Крива Логістичної Регресії')
plt.legend(loc='lower right')
plt.show()

# Вибір ознак для лінійної регресії
X_lin = df[['вік', 'фракція_викиду']]
y_lin = df['сироватковий_креатинін']

# Розділення даних на навчальну та тестову вибірки для лінійної регресії
X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(X_lin, y_lin, test_size=0.2, random_state=42)

# Створення та навчання моделі лінійної регресії
lin_reg = LinearRegression()
lin_reg.fit(X_train_lin, y_train_lin)

# Прогнозування на тестових даних
y_pred_lin = lin_reg.predict(X_test_lin)

# Візуалізація реальних vs прогнозованих значень сироваткового креатиніну
plt.figure()
plt.scatter(y_test_lin, y_pred_lin)
plt.xlabel('Реальний сироватковий креатинін')
plt.ylabel('Прогнозований сироватковий креатинін')
plt.title('Лінійна регресія: Реальне vs Прогнозоване')
plt.show()