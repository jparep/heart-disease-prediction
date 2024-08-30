import pandas as pd
import os

# Merge files
def merge_files(curdir):
    for dirname, _, filenames in os.walk(curdir):
        for filename in filenames:
            print(os.path.join(dirname, filename))
# Load Data
def load_data(file_path):
    return pd.read_csv(file_path)

def main():
    curdir = 'data/raw'
    
    file_path = 'data/raw/heart_2020_cleaned.csv'
    df = load_data(file_path)
    print(df.head())

if __name__=='__main__':
    main()