import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Asegúrate de ajustar la ruta al archivo CSV
csv_path = 'D:\Documents\Fullstack\Intro inteligencia artificial\ProyectoIA\email.csv'
df = pd.read_csv(csv_path)

# Muestra las primeras filas del DataFrame para verificar la carga
print(df.head())

X = df['Message']  # Variable predictora (contenido del correo electrónico)
y = df['Category']  # Variable objetivo (Spam o Ham)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convierte el texto en vectores de características
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Crea un modelo de clasificación Naive Bayes
model = MultinomialNB()

# Entrena el modelo
model.fit(X_train_vectorized, y_train)

# Realiza predicciones en el conjunto de prueba
y_pred = model.predict(X_test_vectorized)

# Calcula la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy}')

# Imprime el informe de clasificación
print("Informe de clasificación:\n", classification_report(y_test, y_pred))

