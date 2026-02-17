from sklearn.preprocessing import LabelEncoder
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

DATA_PATH = "data/processed/processed_gas_sensor.csv"

df = pd.read_csv(DATA_PATH)

X = df.iloc[:, 1:]   # sensors
y = df.iloc[:, 0]    # label column

encoder = LabelEncoder()
y = encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(set(y)), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)

loss, acc = model.evaluate(X_test, y_test)

print("Test Accuracy:", acc)

model.save("models/dl_gas_classifier.keras")
joblib.dump(scaler, "models/scaler.save")
joblib.dump(encoder, "models/label_encoder.save")
