import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def train_and_evaluate_models(X_train, X_test, y_train, y_test, output_dir_plots, output_dir_results, output_dir_models):
    """Trains 4 models, evaluates them, and returns a comparison."""
    os.makedirs(output_dir_plots, exist_ok=True)
    os.makedirs(output_dir_results, exist_ok=True)
    os.makedirs(output_dir_models, exist_ok=True)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results_list = []
    trained_models = {}
    
    print("\nTraining and Evaluating Models...")
    
    for model_name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        trained_models[model_name] = model
        
        # Predict on Test Data
        y_pred = model.predict(X_test)
        
        # We use 'macro' average to treat all 3 classes equally,
        # which is appropriate for a perfectly balanced subset.
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        
        results_list.append({
            'Model': model_name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1
        })
        
        # Generate and save Confusion Matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
        plt.title(f'Confusion Matrix: {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        clean_name = model_name.replace(" ", "_").replace("-", "")
        plt.savefig(os.path.join(output_dir_plots, f'cm_{clean_name}.png'))
        plt.close()
        
    # Create Comparison DataFrame
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(os.path.join(output_dir_results, 'model_comparison.csv'), index=False)
    
    # Select Best Model based on F1 Score
    best_model_row = results_df.loc[results_df['F1 Score'].idxmax()]
    best_model_name = best_model_row['Model']
    best_model = trained_models[best_model_name]
    
    print("\n--- Model Comparison ---")
    print(results_df.to_string(index=False))
    
    print(f"\nFinal Model Selected: {best_model_name}")
    print("Reasoning: It achieved the highest or tied for highest performance across evaluation metrics on the test data.")
    
    # Save the selected final model
    joblib.dump(best_model, os.path.join(output_dir_models, 'final_model.pkl'))
    
    return best_model, best_model_name, results_df
