import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from analisis import cargar_datos
import pickle

np.random.seed(42)

df = cargar_datos("marathon_results_2017.csv")

X = df[['Age', 'Gender_num', '5Kmin', '10Kmin', 'Halfmin']]
y = df["Official Timemin"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
variabilidad = np.std(y_test - y_pred)
std_features = X_train.std()

metricas = {"mae": mae,
            "variabilidad": variabilidad,
            "coef": modelo.coef_.tolist(),
            "std_features": std_features.tolist(),  # <- añadir esto
            "features": ["Edad", "Genero", "5K", "10K", "Media"],
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist()}

with open('modelo_maraton.pkl', 'wb') as f:
    pickle.dump(modelo, f)

with open('metricas_modelo.pkl', 'wb') as f:
    pickle.dump(metricas, f)