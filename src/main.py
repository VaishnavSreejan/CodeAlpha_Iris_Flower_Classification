import os
import joblib
from data_loader import load_and_prepare_data
from preprocessing import explore_and_clean_data, perform_preprocessing
from visualization import create_visualizations
from model_training import train_and_evaluate_models
from prediction import predict_species

def main():
    # Directories
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
    PLOTS_DIR = os.path.join(OUTPUTS_DIR, 'plots')
    RESULTS_DIR = os.path.join(OUTPUTS_DIR, 'results')
    MODELS_DIR = os.path.join(OUTPUTS_DIR, 'models')
    
    print("="*60)
    print(" IRIS FLOWER CLASSIFICATION PIPELINE ".center(60, "="))
    print("="*60)
    
    # 1. Load Dataset
    df = load_and_prepare_data()
    
    # 2. Explore and Clean
    df = explore_and_clean_data(df)
    
    # 3. Create Visualizations
    create_visualizations(df, PLOTS_DIR)
    
    # 4. Preprocess Data (Split & Scale)
    X_train, X_test, y_train, y_test, scaler = perform_preprocessing(df)
    
    # Save the scaler so it can be used independently later
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    
    # 5. Train & Evaluate Models
    best_model, best_model_name, results_df = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, PLOTS_DIR, RESULTS_DIR, MODELS_DIR
    )
    
    # 6. Make a Sample Prediction
    print("\n--- Testing Sample Prediction ---")
    sl, sw, pl, pw = 5.1, 3.5, 1.4, 0.2
    print(f"Input Features: sepal_length={sl}, sepal_width={sw}, petal_length={pl}, petal_width={pw}")
    
    pred = predict_species(best_model, scaler, sl, sw, pl, pw)
    print(f"Predicted Species: {pred}")
    print("Note: The sample features correspond to Iris setosa.")
    
    print("\n" + "="*60)
    print(" PIPELINE COMPLETED SUCCESSFULLY ".center(60, "="))
    print("="*60)

if __name__ == "__main__":
    main()
