# Import libraries
import pandas as pd
from sklearn.experimental import enable_iterative_imputer # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import logging

# Global configurations
DATA_FILE_PATH = 'data/prossed/merged_heart_data.csv'
TARGET_COLUMN = 'HeartDisease'
TEST_SIZE = 0.2
RANDOM_STATE = 12

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()])

def load_data(file_path):
    """Load data from CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}")
        return df
    except Exception  as e:
        logging.info(f"Failed to load data {e}")
        raise

def clean_data(df, target_column):
    """Clean data by removing irrelevant and all-NA columns and splitting target."""
    df = df.drop(columns=["Unnamed: 0"], errors='ignore') 
    df = df.dropna(subset=[target_column]) # Drop rows where target is NaN
    df = df.dropna(axis=1, how='all')
    return df.drop(target_column, axis=1), df[target_column]

def create_pipeline(imputer, transformer):
    pipe = Pipeline(steps=[
        ('imputer', imputer),
        ('transfomer', transformer)
    ])