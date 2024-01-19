import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

# Asegúrate de ajustar la ruta al archivo CSV
csv_path = 'D:\Documents\Fullstack\Intro inteligencia artificial\ProyectoIA\email.csv'
df = pd.read_csv(csv_path)

# Supongamos que 'feature_columns' son las columnas de características y 'target_column' es la variable objetivo
feature_columns = ['Message']
target_column = 'Category'

# Separa las características (X) y la variable objetivo (y)
X = df[feature_columns]
y = df[target_column]

# Inicializa el vectorizador TF-IDF
tfidf_vectorizer = TfidfVectorizer()

# Aplica el vectorizador a la columna 'Message'
message_tfidf = tfidf_vectorizer.fit_transform(X['Message'])

# Convierte el resultado en un DataFrame y concaténalo con las características existentes
X_encoded = pd.concat([X, pd.DataFrame(message_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())], axis=1)

# Divide los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Identifica las columnas numéricas
numeric_columns = X_encoded.select_dtypes(include=['float64', 'int64']).columns

# Normaliza solo las columnas numéricas (opcional, dependiendo del modelo)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numeric_columns])
X_test_scaled = scaler.transform(X_test[numeric_columns])

# Entrena el modelo
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

# Realiza predicciones en el conjunto de prueba
y_pred = model.predict(X_test_scaled)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy}')

# Imprimir informe de clasificación
print("Informe de clasificación:\n", classification_report(y_test, y_pred))
