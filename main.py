import pandas as pd

# Load Data
def load_data(file_path):
    return pd.read_csv(file_path)

def main():
    
    # Load data
    file_path = 'data/processed/merged_heart_data.csv'
    df = load_data(file_path)
    print(df.head())

if __name__=='__main__':
    main()