import pandas as pd

# Cargar el dataset
df = pd.read_csv("C:/Users/USUARIO/Desktop/Caso de Estudio Analisis y Predicción de Datos de Empleados/EmployeesData.csv")

print(df.head())

# Verificar valores faltantes
print(df.isnull().sum())

# Convertir 0 y 1 a Not Leave y Leave respectivamente
df['LeaveOrNot'] = df['LeaveOrNot'].map({0: 'Not Leave', 1: 'Leave'})
df.dropna(subset=['ExperienceInCurrentDomain', 'JoiningYear'], inplace=True)

# Imputar Age con la media
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Imputar PaymentTier con la moda
mode_payment_tier = df['PaymentTier'].mode()[0]
df['PaymentTier'].fillna(mode_payment_tier, inplace=True)
from scipy import stats

# Calcula el IQR para cada columna y filtra los valores atípicos
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

import matplotlib.pyplot as plt

df['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Distribución de Género')
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Histograma
df['Education'].value_counts().plot(kind='bar', ax=ax[0])
ax[0].set_title('Histograma de Educación')

# Gráfico de torta
df['Education'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax[1])
ax[1].set_title('Distribución de Educación')

plt.tight_layout()
plt.show()

plt.hist(df[df['LeaveOrNot'] == 'Leave']['Age'], bins=10, alpha=0.5, label='Leave')
plt.hist(df[df['LeaveOrNot'] == 'Not Leave']['Age'], bins=10, alpha=0.5, label='Not Leave')
plt.legend()
plt.title('Distribución de Edad por LeaveOrNot')
plt.show()

df['LeaveOrNot'].value_counts().plot(kind='bar')
plt.title('Distribución de LeaveOrNot')
plt.show()

from sklearn.model_selection import train_test_split

# Variables dummies para categóricas
df_dummies = pd.get_dummies(df.drop('LeaveOrNot', axis=1))

# Variable objetivo
y = df['LeaveOrNot']

# Partición estratificada
X_train, X_test, y_train, y_test = train_test_split(df_dummies, y, test_size=0.2, stratify=y, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Entrenar Random Forest sin ajuste de peso de clases
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Predicciones
y_pred_train = rf_classifier.predict(X_train)
y_pred_test = rf_classifier.predict(X_test)

# Accuracy
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

print(f"Accuracy (Entrenamiento): {accuracy_train:.4f}")
print(f"Accuracy (Test): {accuracy_test:.4f}")

# Entrenar Random Forest con ajuste de peso de clases
rf_classifier_balanced = RandomForestClassifier(random_state=42, class_weight="balanced")
rf_classifier_balanced.fit(X_train, y_train)

# Predicciones
y_pred_train_balanced = rf_classifier_balanced.predict(X_train)
y_pred_test_balanced = rf_classifier_balanced.predict(X_test)

# Accuracy
accuracy_train_balanced = accuracy_score(y_train, y_pred_train_balanced)
accuracy_test_balanced = accuracy_score(y_test, y_pred_test_balanced)

print(f"Accuracy (Entrenamiento, Balanced): {accuracy_train_balanced:.4f}")
print(f"Accuracy (Test, Balanced): {accuracy_test_balanced:.4f}")

from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

# F1 Score
f1_score_test = f1_score(y_test, y_pred_test, pos_label="Leave")
f1_score_test_balanced = f1_score(y_test, y_pred_test_balanced, pos_label="Leave")

print(f"F1 Score (Test): {f1_score_test:.4f}")
print(f"F1 Score (Test, Balanced): {f1_score_test_balanced:.4f}")

# Matriz de Confusión
cm = confusion_matrix(y_test, y_pred_test, labels=rf_classifier.classes_)
cm_balanced = confusion_matrix(y_test, y_pred_test_balanced, labels=rf_classifier_balanced.classes_)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ConfusionMatrixDisplay(cm, display_labels=rf_classifier.classes_).plot(ax=ax[0])
ax[0].set_title('Matriz de Confusión (Sin Ajuste)')

ConfusionMatrixDisplay(cm_balanced, display_labels=rf_classifier_balanced.classes_).plot(ax=ax[1])
ax[1].set_title('Matriz de Confusión (Con Ajuste)')

plt.tight_layout()
plt.show()

# Seleccionar solo columnas numéricas para el cálculo del IQR
df_numeric = df.select_dtypes(include=['number'])

# Calcula el IQR solo para las columnas numéricas
Q1 = df_numeric.quantile(0.25)
Q3 = df_numeric.quantile(0.75)
IQR = Q3 - Q1

# Filtrar los valores atípicos basándose en el IQR de las columnas numéricas
# Primero, crea una máscara para identificar las filas con valores atípicos
mask = df_numeric.apply(lambda x: ~((x < (Q1 - 1.5 * IQR)) | (x > (Q3 + 1.5 * IQR))).any(axis=1), axis=0)

# Aplicar la máscara al dataframe original para excluir valores atípicos
df = df[mask]

"""Nota: Este paso puede requerir ajustes adicionales dependiendo de cómo
pandas maneja la aplicación de la máscara a df, dado que mask se calculó
solo sobre df_numeric. Asegúrate de que la longitud de 'mask' coincida
con el número de filas en 'df' para evitar errores."""

