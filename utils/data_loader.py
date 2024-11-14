import pandas as pd

def load_data(file_path):
    # Load CSV data into a pandas DataFrame
    data = pd.read_csv(file_path)
    return data
