import pandas as pd
import tensorflow as tf
import joblib
from sklearn.metrics import classification_report, mean_squared_error

DATA_PATH = "data/processed/processed_gas_sensor.csv"

df = pd.read_csv(DATA_PATH)

# ---------- DL MODEL ----------
model = tf.keras.models.load_model("models/dl_gas_classifier.keras")
scaler = joblib.load("models/scaler.save")
encoder = joblib.load("models/label_encoder.save")

X = df.iloc[:, 1:]
y = encoder.transform(df.iloc[:, 0])

X_scaled = scaler.transform(X)

predictions = model.predict(X_scaled)
predicted_classes = predictions.argmax(axis=1)

print("Deep Learning Classification Report:")
print(classification_report(y, predicted_classes))

# ---------- TIME SERIES ----------
ts_model = joblib.load("models/timeseries_model.pkl")

sensor_col = df.columns[1]

df['future'] = df[sensor_col].shift(-1)
df.dropna(inplace=True)

X_ts = df[[sensor_col]]
y_ts = df['future']

preds = ts_model.predict(X_ts)

mse = mean_squared_error(y_ts, preds)

print("Time Series MSE:", mse)
