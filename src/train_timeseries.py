import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

DATA_PATH = "data/processed/processed_gas_sensor.csv"

df = pd.read_csv(DATA_PATH)

# choose one sensor for forecasting
sensor_col = df.columns[1]  # sensor_0 (since column 0 is label)

# create future target
df['future'] = df[sensor_col].shift(-1)

df.dropna(inplace=True)

X = df[[sensor_col]]
y = df['future']

# DO NOT shuffle -> keeps time order
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

score = model.score(X_test, y_test)

print("Time Series R2:", score)

joblib.dump(model, "models/timeseries_model.pkl")
