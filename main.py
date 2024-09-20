# Import libraries
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
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
    
def get_data_types(X: pd.DataFrame) -> Tuple[pd.Index, pd.Index]:
    """Get numerical and categorical column names."""
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols= X.select_dtypes(include=['object']).columns
    return num_cols, cat_cols

def create_preprocessing_pipeline(num_cols: pd.Index, cat_cols: pd.Index) -> ColumnTransformer:
    """Build the preprocessing pipeline for numerical and categorical columns."""
    num_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    return ColumnTransformer(transformers=[
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])

def build_model_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    """Combine preprocessing and model into a pipeline."""
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
    ])

def hyperparameter_tuning(model_pipeline: Pipeline, X_train: pd. DataFrame, y_train: pd.Series) -> Pipeline:
    """Tune hyperparameters for Randomforest classsifier into a pipeline."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(estimator=model_pipeline,
                               param_grid=param_grid,
                               cv=cv,
                               n_jobs=-1,
                               scoring='f1',
                               verbose=1)
    logging.info('Starting hyperparameter tuning using GridSearchCV.')
    grid_search.fit(X_train, y_train)
    
    logging.info(f'Best parameters: {grid_search.best_params_}')
    logging.info(f'Best score: {grid_search.best_score_}')
    return grid_search.best_estimator_


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Evaluate model performance on the test sets."""
    try:
        y_pred = model.predict(X_test)
        score_dict = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F-1 Score': f1_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_pred)
        }
        logging.info(f'Model evaluation scores: {score_dict}')
        report = classification_report(y_test, y_pred)
        logging.info(f'Classification Report: {report}')
        
        print("Evaluation Metrics:")
        for key, value in score_dict.items():
            print(f'{key}: {value:.4f}')
            
        print('\n\nClassfification Report:\n{report}')
    except Exception as e:
        logging.error(f'Error during model evaluation: {e}')
        raise

def save_model(model: Pipeline, file_name: str) -> None:
    """Save the trained model to disk."""
    try:
        joblib.dump(model, file_name)
        logging.info(f"Model saved to {file_name}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

def load_model(file_name: str) -> Pipeline:
    """Load a saved model from disk."""
    try:
        model = joblib.load(file_name)
        logging.info(f'Mode loaded from {file_name}')
        return model
    except FileNotFoundError as e:
        logging.error(f"Model file not found: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def main():
    # Load the data file into DataFrame
    df = load_data(DATA_FILE_PATH)
    
    # Clean DataFrame and separate features (X) and target (y) variables
    X, y = clean_data(df)
    
    # Separate into numerical and categorical columns from features (X)
    num_cols, cat_cols = get_data_types(X)
    
    # Create Pipeline of numerical and categorical features into ColumnTransformer
    preprocessor = create_preprocessing_pipeline(num_cols, cat_cols)
    
    # Build preprocessor model pipeline
    model_pipeline = build_model_pipeline(preprocessor)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # Hyperparameter tune the model pipeline
    best_model = hyperparameter_tuning(model_pipeline, X_train, y_train)
    
    # Evaludate the model performance
    evaluate_model(best_model)