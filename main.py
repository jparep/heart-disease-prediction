import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import logging

# Global configurations
DATA_FILE_PATH = 'data/processed/merged_heart_data.csv'
TARGET_COLUMN = 'HeartDisease'
TEST_SIZE = 0.2
RANDOM_STATE = 12

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                    handlers=[logging.FileHandler("app.log"),
                              logging.StreamHandler()])

def load_data(file_path):
    """Load data from CSV file."""
    try:
        logging.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise

def clean_data(df, target_column):
    """Clean data by removing irrelevant and all-NA columns and splitting target."""
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')  # Drop irrelevant index column
    df = df.dropna(subset=[target_column])  # Drop rows where target is NaN
    df = df.dropna(axis=1, how='all')  # Drop columns with all NaN values
    logging.info(f"Data cleaned, all-NaN columns removed, and target column '{target_column}' separated.")
    return df.drop(target_column, axis=1), df[target_column]

def get_column_types(df):
    """Identify categorical and numerical columns in the feature set."""
    categorical_columns = df.select_dtypes(include=['object']).columns
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    logging.info(f"Identified {len(categorical_columns)} categorical and {len(numerical_columns)} numerical columns.")
    return categorical_columns, numerical_columns

def create_pipeline(imputer, transformer):
    """Create a generic pipeline with imputation and transformation."""
    return Pipeline(steps=[
        ('imputer', imputer),
        ('transformer', transformer)
    ])

def create_preprocessing_pipeline(cat_cols, num_cols):
    """Build the preprocessing pipeline for both categorical and numerical columns."""
    num_pipeline = create_pipeline(IterativeImputer(), StandardScaler())
    cat_pipeline = create_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    
    logging.info("Preprocessing pipeline created.")
    return preprocessor

def build_model_pipeline(preprocessor):
    """Combine preprocessing and the RandomForest model into a pipeline."""
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(random_state=RANDOM_STATE))
    ])
    logging.info("Model pipeline with RandomForest created.")
    return model_pipeline

def train_and_evaluate_pipeline(model_pipeline, X_train, y_train):
    """Fit the model pipeline to the training data and evaluate with cross-validation."""
    logging.info("Fitting the model pipeline to the training data.")
    model_pipeline.fit(X_train, y_train)
    
    # Perform cross-validation for evaluation
    cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=5)
    logging.info(f"Cross-validation scores: {cv_scores}")
    logging.info(f"Mean CV score: {cv_scores.mean()}")

def main():
    """Main function to execute the data pipeline and model training."""
    # Step 1: Load the data
    df = load_data(DATA_FILE_PATH)
    
    # Step 2: Clean data and split into features (X) and target (y)
    X, y = clean_data(df, TARGET_COLUMN)
    
    # Step 3: Identify categorical and numerical columns
    categorical_columns, numerical_columns = get_column_types(X)
    
    # Step 4: Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(categorical_columns, numerical_columns)
    
    # Step 5: Build full model pipeline
    model_pipeline = build_model_pipeline(preprocessor)
    
    # Step 6: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # Step 7: Train and evaluate the model pipeline
    train_and_evaluate_pipeline(model_pipeline, X_train, y_train)

if __name__ == '__main__':
    main()
