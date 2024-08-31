import pandas as pd
from sklearn.experimental import enable_iterative_imputer # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

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