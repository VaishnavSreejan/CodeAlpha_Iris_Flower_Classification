import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def explore_and_clean_data(df):
    """Performs required Exploratory Data Analysis and Cleaning."""
    print("\n--- Missing Values ---")
    missing_values = df.isnull().sum()
    print(missing_values)
    
    if missing_values.sum() == 0:
        print("Explicitly reporting: There are NO missing values in the dataset.")
        
    print("\n--- Duplicate Rows ---")
    duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows found: {duplicates}")
    
    if duplicates > 0:
        print("Removing duplicate rows to ensure clean logic...")
        df = df.drop_duplicates()
        
    return df

def perform_preprocessing(df):
    """Splits data and applies feature scaling."""
    # Separate input features (X) and target (y)
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df['species']
    
    # Train-test splitting is necessary to separate our dataset into a part that the model learns from (train), 
    # and a separate unseen part (test) to evaluate how well the model generalizes to new data.
    # We use stratification to maintain the proportions of the target class in both splits.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set size: {X_train.shape[0]} rows")
    print(f"Test set size: {X_test.shape[0]} rows")
    
    # Feature scaling is performed because many spatial algorithms (like KNN and Logistic Regression)
    # perform better when numerical features share a common scale and distribution, preventing 
    # larger values (like sepal_length) from overpowering smaller values (like petal_width).
    scaler = StandardScaler()
    
    # IMPORTANT: Fit the scaler ONLY on the training data to prevent data leakage,
    # then transform both the training data and the test data.
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
