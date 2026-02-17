import pandas as pd
import os

RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed/processed_gas_sensor.csv"


def parse_sparse_row(row):
    label = int(row[0])  # first column is gas class
    
    features = [0] * 128  # dataset has 128 features
    
    for item in row[1:]:
        if pd.isna(item):
            continue
            
        index, value = item.split(":")
        features[int(index) - 1] = float(value)

    return [label] + features


def load_data():
    parsed_rows = []

    for file in os.listdir(RAW_PATH):
        if file.endswith(".dat"):
            path = os.path.join(RAW_PATH, file)

            df = pd.read_csv(
                path,
                sep=" ",
                header=None
            )

            for _, row in df.iterrows():
                parsed_rows.append(parse_sparse_row(row))

    columns = ["label"] + [f"sensor_{i}" for i in range(128)]

    final_df = pd.DataFrame(parsed_rows, columns=columns)

    return final_df


def save_data(df):
    df.to_csv(PROCESSED_PATH, index=False)
    print("Processed data saved.")


if __name__ == "__main__":
    df = load_data()
    save_data(df)
