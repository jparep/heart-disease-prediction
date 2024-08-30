import pandas as pd

# Load Data
def load_data(file_path):
    return pd.read_csv(file_path)

def main():
    file_path = 'data/raw/heart_2020_cleaned.csv'
    df = load_data(file_path)

if __name__=='__main__':
    main()