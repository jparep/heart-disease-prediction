# Import libraries
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from typing import Tuple, Dict, Any
import logging
import os
import joblib

# Global configuration (can be overriden by environment vairables)
DATA_FILE_PATH = os.getenv('DATA_FILE_PATH', 'data/preprocessed/merged_heart_data.csv')
TARGET_COLUMN = os.getenv('TARGET_COLUMN', 'HeartDisease')
TEST_SIZE = float(os.getenv('TEST_SIZE', 0.2))
RANDOM_STATE = int(os.getenv('RANDOM_STATE', 12))
MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH', 'models/model.joblib')

# Hyperparamenter grid for RandomFOrest
param_grid = {
    'model_n_estimators': [50, 100, 200],
    'model_max_depth': [None, 10, 20, 30],
    'model_min_samples_split': [2, 5, 10],
    'model_min_samples_leaf': [1, 2, 4],
    'model_bootstrap': [True, False]
}

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(astime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()])

def load_data(file_path: str) -> pd.DataFrame:
    """Load data ffrom the specified CSV fiel."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f'Data loaded successfully from {file_path}')
        return df
    except FileExistsError as e:
        logging.error(f'Data fiel is empty: {e}')
        raise
    except pd.errors.EmptyDataError as e:
        logging.error(f'Failed to load data: {e}')
        raise
    except Exception as e:
        logging.error(f'Failed to load data: {e}')
        raise

def clean_data(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Clean data and split target column."""
    df = df.drop(columns=['Unnamed: 0'], error='ignore') # Drop the 'Unnamed: 0' column if it exist
    df = df.dropna(subset=[target_column]) # Remove all the ROWS where the target column is NA
    df = df.dropna(axis=1, how='all') # Remove COLUMNS where all the values are NA
    return df.drop(target_column, axis=1), df[target_column]
    