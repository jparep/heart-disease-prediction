import pandas as pd
from sklearn.experimental import enable_iterative_imputer # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import logging



def load_data(file_path):
    """Load Data from csv file"""
    return pd.read_csv(file_path)

def convert_cat_to_num(df):
    pass
    

def main():
    
    # Load data
    file_path = 'data/processed/merged_heart_data.csv'
    df = load_data(file_path)
    
    # Drop unncessary columns
    df = df.drop('Unnamed: 0', axis=1)
    
    # Drop na from target variable
    df = df.dropna(subset=['HeartDisease'])
    print(f"Total NA for target varible = {df['HeartDisease'].isna().sum()}")

    print(df.columns)
if __name__=='__main__':
    main()