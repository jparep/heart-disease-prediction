# Import libraries
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, classification_report, accuracy_score, 
                             f1_score, precision_score, recall_score)
import logging
from typing import Tuple, Dict, Any
import os

# Global configuration (can be overridden by environment variables)
DATA_FILE_PATH = os.getenv('DATA_FILE_PATH', 'data/processed/merged_heart_data.csv')
TARGET_COLUMN = os.getenv('TARGET_COLUMN', 'HeartDisease')
TEST_SIZE = float(os.getenv('TEST_SIZE', 0.2))
RANDOM_STATE = int(os.getenv('RANDOM_STATE', 12))

# Hyperparameter grid for RandomForest
param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__bootstrap': [True, False]
}

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()])

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from the specified CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}")
        return df
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logging.error(f"Data file is empty: {e}")
        raise
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise

def clean_data(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Clean data and split target column."""
    df = df.drop(columns=["Unnamed: 0"], errors='ignore')
    df = df.dropna(subset=[target_column])
    df = df.dropna(axis=1, how='all')
    return df.drop(target_column, axis=1), df[target_column]

def get_column_types(df: pd.DataFrame) -> Tuple[pd.Index, pd.Index]:
    """Get numerical and categorical column names."""
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    return num_cols, cat_cols

def create_preprocessing_pipeline(num_cols: pd.Index, cat_cols: pd.Index) -> ColumnTransformer:
    """Build the preprocessing pipeline for numerical and categorical columns."""
    num_pipeline = Pipeline(steps=[
        ('imputer', IterativeImputer()),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    return ColumnTransformer(transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

def build_model_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    """Combine preprocessing and model into a pipeline."""
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
    ])

def hyperparameter_tuning(model_pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Pipeline, Dict[str, Any], Dict[str, Any]]:
    """Tune hyperparameters for RandomForest in the pipeline."""
    grid_search = GridSearchCV(estimator=model_pipeline,
                               param_grid=param_grid,
                               cv=5,
                               n_jobs=-1,
                               scoring='f1',
                               verbose=1)

    logging.info("Starting hyperparameter tuning using GridSearchCV.")
    grid_search.fit(X_train, y_train)

    logging.info(f"Best parameters: {grid_search.best_params_}")
    logging.info(f"Best score: {grid_search.best_score_}")

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_

def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Dict[str, float], str]:
    """Evaluate model performance on the test set."""
    try:
        y_pred = model.predict(X_test)

        scores = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_pred)
        }

        logging.info(f"Model evaluation scores: {scores}")
        report = classification_report(y_test, y_pred)
        logging.info(f"Classification report:\n{report}")

        print("Evaluation Metrics:")
        for metric, score in scores.items():
            print(f"{metric}: {score:.4f}")
        print("\nClassification Report:")
        print(report)

        return scores, report

    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise

def main():
    """Main function to load data, preprocess, train, and evaluate the model."""
    df = load_data(DATA_FILE_PATH)
    X, y = clean_data(df, TARGET_COLUMN)
    num_cols, cat_cols = get_column_types(X)
    preprocessor = create_preprocessing_pipeline(num_cols, cat_cols)
    model_pipeline = build_model_pipeline(preprocessor)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    model, model_params, model_cv = hyperparameter_tuning(model_pipeline, X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == '__main__':
    main()
