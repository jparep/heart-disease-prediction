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

# Global configurations
DATA_FILE_PATH = 'data/processed/merged_heart_data.csv'
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
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise

def clean_data(df, target_column):
    """Clean data by removing irrelevant and all-NA columns and splitting target."""
    df = df.drop(columns=["Unnamed: 0"], errors='ignore') 
    df = df.dropna(subset=[target_column])  # Drop rows where target is NaN
    df = df.dropna(axis=1, how='all')  # Drop columns with all NaN values
    return df.drop(target_column, axis=1), df[target_column]

def get_column_types(df):
    """Get numerical and categorical column names."""
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    return num_cols, cat_cols

def create_pipeline(imputer, transformer):
    """Build a generic preprocessing pipeline."""
    return Pipeline(steps=[
        ('imputer', imputer),
        ('transformer', transformer)
    ])

def create_preprocessing_pipeline(num_cols, cat_cols):
    """Build the preprocessing pipeline for both categorical and numerical columns."""
    num_pipeline = create_pipeline(IterativeImputer(), StandardScaler())
    cat_pipeline = create_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))
    
    return ColumnTransformer(transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

def build_model_pipeline(preprocessor):
    """Combine preprocessing and the model into a pipeline."""
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(random_state=RANDOM_STATE))
    ])

def hyperparameter_tuning(model_pipeline, X_train, y_train):
    """Tune the hyperparameters for the RandomForest model in the pipeline."""
    try:
        # Define the hyperparameters to tune
        param_grid = {
            'model__n_estimators': [50, 100, 200],  # Number of trees in the forest
            'model__max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
            'model__min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
            'model__min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
            'model__bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
        }

        # Initialize GridSearchCV with cross-validation, tuning the full pipeline
        grid_search = GridSearchCV(
            estimator=model_pipeline,
            param_grid=param_grid,
            cv=5,  # 5-fold cross-validation
            n_jobs=-1,  # Use all available cores
            scoring='accuracy',  # Evaluation metric, can be changed to 'f1', 'roc_auc', etc.
            verbose=1  # Enable verbosity to track progress
        )

        logging.info("Starting hyperparameter tuning using GridSearchCV.")
        
        # Fit GridSearchCV
        grid_search.fit(X_train, y_train)
        
        # Logging the best parameters and score
        logging.info(f"Best parameters: {grid_search.best_params_}")
        logging.info(f"Best score: {grid_search.best_score_}")

        # Return the best estimator (pipeline) and all results
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_

    except Exception as e:
        logging.error(f"Error occurred during hyperparameter tuning: {e}")
        raise

def train_model(model, X_train, y_train):
    """Fit the model pipeline to the training data."""
    return model.fit(X_train, y_train)

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on the test set."""
    try:
        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate performance metrics
        scores = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='binary'),
            'Recall': recall_score(y_test, y_pred, average='binary'),
            'F1 Score': f1_score(y_test, y_pred, average='binary'),
            'ROC AUC': roc_auc_score(y_test, y_pred)  # For binary classification
        }
        
        # Log the results
        logging.info(f"Model evaluation scores: {scores}")
        
        # Print detailed classification report
        report = classification_report(y_test, y_pred)
        logging.info(f"Classification report:\n{report}")

        # Display results in a structured manner
        print("Evaluation Metrics:")
        for metric, score in scores.items():
            print(f"{metric}: {score:.4f}")
        print("\nDetailed Classification Report:")
        print(report)

        return scores, report  # Return for further use if needed

    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise

def main():
    """Main function to load data, preprocess, train, and evaluate the model."""
    # Load data from data file
    df = load_data(DATA_FILE_PATH)
    
    # Clean data and separate into target (y) and features (X)
    X, y = clean_data(df, TARGET_COLUMN)
    
    # Separate numerical and categorical columns
    num_cols, cat_cols = get_column_types(X)
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(num_cols, cat_cols)
    
    # Build the model pipeline
    model_pipeline = build_model_pipeline(preprocessor)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # Tune the model hyperparameters
    model, model_params, model_cv = hyperparameter_tuning(model_pipeline, X_train, y_train)
    
    # Evaluate model performance
    evaluate_model(model, X_test, y_test)

if __name__ == '__main__':
    main()
