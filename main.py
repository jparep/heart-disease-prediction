import pandas as pd
import os

# Merge files
def merge_files(data_dir):
    for dirname, _, filenames in os.walk(data_dir):
        for filename in filenames:
            print(os.path.join(dirname, filename))
# Load Data
def load_data(file_path):
    return pd.read_csv(file_path)

def main():
    data_dir = 'data/raw/'
    file_names = ['heart_2020_cleaned.csv', 'heart_2022_n0_nans_csv','heart_2022_with_nans.csv']
    merge_files(data_dir)
    
    file_path = 'data/raw/heart_2020_cleaned.csv'
    df = load_data(file_path)
    print(df.head())

if __name__=='__main__':
    main()