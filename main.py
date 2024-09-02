import pandas as pd
from sklearn.experimental import enable_iterative_imputer # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import logging
 
 # Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(level)s - %(message)s', 
                    handlers = [logging.FileHandler("app.log"),
                                logging.StreamHandler()])


def load_data(file_path):
    """Load Data from csv file"""
    return pd.read_csv(file_path)

def remove_all_nan_cols(df):
    """Remove columns with all NA and target value with NAN"""
    df = df.drop('Unnamed: 0', axis=1)
    df = df.dropna(subset=['HeartDisease'])
    df = df.dropna(axis=1, how='all')
    return df

def separate_to_cat_num_cols(df):
    """Separate features into categorical and numerical columns"""
    cat_cols = df.select_dtypes(include=['objects']).columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    return cat_cols, num_cols

def numerical_pipeline(df):
    """Transformer Pipeline for Numerical features"""
    num_pipe = Pipeline(steps=[
        'imputer', IterativeImputer(),
        'scaler', StandardScaler()
    ])
    return num_pipe

def categorical_pipeline(df):
    "Pipeline transformer for categorical features"
    cat_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer()),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    

def main():
    
    # Load data
    file_path = 'data/processed/merged_heart_data.csv'
    df = load_data(file_path)
    
    # Preprocess data
    df =remove_all_nan_cols(df)
    cat_cols, num_cols = separate_to_cat_num_cols(df)
    
    
if __name__=='__main__':
    main()