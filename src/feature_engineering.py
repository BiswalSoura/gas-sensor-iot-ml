import pandas as pd

DATA_PATH = "data/processed/processed_gas_sensor.csv"

def prepare_features():
    df = pd.read_csv(DATA_PATH)

    # Assume last column is label (gas type)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return X, y


if __name__ == "__main__":
    X, y = prepare_features()
    print("Features ready:", X.shape)
