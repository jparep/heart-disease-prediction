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

def load_data(file_path):
    """Load Data from CSV file."""
    try:
        logging.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise

def remove_all_nan_cols(df):
    """Remove columns with all NA and drop target values with NAN."""
    df = df.drop('Unnamed: 0', axis=1)
    df = df.dropna(subset=['HeartDisease'])
    df = df.dropna(axis=1, how='all')
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    logging.info("Columns with all NaNs removed.")
    return X,y

def separate_to_cat_num_cols(X):
    """Separate features into categorical and numerical columns."""
    cat_cols = X.select_dtypes(include=['object']).columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    logging.info(f"Categorical columns: {cat_cols}")
    logging.info(f"Numerical columns: {num_cols}")
    return cat_cols, num_cols

def numerical_pipeline():
    """Transformer Pipeline for Numerical features."""
    num_pipe = Pipeline(steps=[
        ('imputer', IterativeImputer()),
        ('scaler', StandardScaler())
    ])
    return num_pipe

def categorical_pipeline():
    """Pipeline transformer for categorical features."""
    cat_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    return cat_pipe

def preprocessing(cat_cols, num_cols):
    """Preprocessing step with ColumnTransformer."""
    preprocessor = ColumnTransformer(transformers=[
        ('cat', categorical_pipeline(), cat_cols),
        ('num', numerical_pipeline(), num_cols)
    ])
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(random_state=12))
    ])
    logging.info("Preprocessing and model pipeline created.")
    return model_pipeline

def main():
    file_path = 'data/processed/merged_heart_data.csv'
    
    # Load data
    df = load_data(file_path)
    
    # Preprocess data
    X, y = remove_all_nan_cols(df)
    cat_cols, num_cols = separate_to_cat_num_cols(X)
    
    # Create preprocessing and model pipeline
    model_pipeline = preprocessing(cat_cols, num_cols)
    
    # Split train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, randome_state=12)
    
    # Model training can be added here
    model_pipeline.fit(X_train, y_train)
    
if __name__ == '__main__':
    main()
