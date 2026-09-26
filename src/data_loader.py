import pandas as pd
from sklearn.datasets import load_iris

def load_and_prepare_data():
    """Loads the Iris dataset from scikit-learn and prepares a DataFrame."""
    print("Loading dataset from sklearn.datasets.load_iris()...")
    iris = load_iris()
    
    # Create DataFrame with clear feature names
    df = pd.DataFrame(iris.data, columns=[
        'sepal_length', 'sepal_width', 'petal_length', 'petal_width'
    ])
    
    # Add target feature
    df['species'] = iris.target
    
    # Map numerical target values to string names
    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    df['species'] = df['species'].map(species_map)
    
    # Required Outputs from Step 3
    print("\n--- First 5 rows ---")
    print(df.head())
    
    print("\n--- Last 5 rows ---")
    print(df.tail())
    
    print(f"\n--- Shape --- \nRows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    print("\n--- Column Names ---")
    print(list(df.columns))
    
    print("\n--- Data Types ---")
    print(df.dtypes)
    
    print("\n--- Descriptive Statistics ---")
    print(df.describe())
    
    print("\n--- Target/Class Distribution ---")
    print(df['species'].value_counts())
    
    return df
