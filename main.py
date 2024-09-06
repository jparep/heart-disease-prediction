import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                    handlers=[logging.FileHandler("app.log"),
                              logging.StreamHandler()])

def load_csv_data(file_path):
    """Load data from a CSV file."""
    try:
        logging.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise

def clean_and_split_target(df, target_column='HeartDisease'):
    """Remove all-NA columns and separate features from the target column."""
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')  # Drop irrelevant index column
    df = df.dropna(subset=[target_column])  # Drop rows where target is NaN
    df = df.dropna(axis=1, how='all')  # Drop columns with all NaN values
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    logging.info(f"Cleaned data, removed all-NaN columns, and split target: {target_column}")
    return X, y

def get_categorical_and_numerical_columns(X):
    """Identify categorical and numerical columns in the feature set."""
    categorical_columns = X.select_dtypes(include=['object']).columns
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
    
    logging.info(f"Categorical columns: {categorical_columns}")
    logging.info(f"Numerical columns: {numerical_columns}")
    return categorical_columns, numerical_columns

def create_numerical_pipeline():
    """Create a pipeline for processing numerical features."""
    num_pipeline = Pipeline(steps=[
        ('imputer', IterativeImputer()),  # Impute missing values using IterativeImputer
        ('scaler', StandardScaler())  # Standardize the data
    ])
    return num_pipeline

def create_categorical_pipeline():
    """Create a pipeline for processing categorical features."""
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with the most frequent value
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical data
    ])
    return cat_pipeline

def create_preprocessing_pipeline(categorical_columns, numerical_columns):
    """Build the preprocessing pipeline using both categorical and numerical pipelines."""
    preprocessor = ColumnTransformer(transformers=[
        ('num', create_numerical_pipeline(), numerical_columns),  # Apply numerical pipeline
        ('cat', create_categorical_pipeline(), categorical_columns)  # Apply categorical pipeline
    ])
    
    # Create full pipeline including the model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(random_state=12))  # Use RandomForest for classification
    ])
    
    logging.info("Preprocessing and model pipeline created.")
    return model_pipeline

def main():
    # File path for the dataset
    file_path = 'data/processed/merged_heart_data.csv'
    
    # Step 1: Load the data
    df = load_csv_data(file_path)
    
    # Step 2: Clean data and split into features (X) and target (y)
    X, y = clean_and_split_target(df)
    
    # Step 3: Identify categorical and numerical columns
    categorical_columns, numerical_columns = get_categorical_and_numerical_columns(X)
    
    # Step 4: Create preprocessing and model pipeline
    model_pipeline = create_preprocessing_pipeline(categorical_columns, numerical_columns)
    
    # Step 5: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
    
    # Step 6: Fit the model pipeline on training data
    logging.info("Fitting the model pipeline to the training data.")
    model_pipeline.fit(X_train, y_train)
    
    # Optionally, add model evaluation here if needed (e.g., cross-validation or test evaluation)
    cross_val_score(model_pipeline, X_train, y_train, cv=5, random_state=12)
    
    logging.info("Model pipeline training complete.")

if __name__ == '__main__':
    main()
