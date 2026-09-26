import pandas as pd

def predict_species(model, scaler, sepal_length, sepal_width, petal_length, petal_width):
    """
    Reusable prediction function taking 4 numerical measurements
    and returning the textual species prediction.
    """
    # 1. Format input as a Pandas DataFrame with exact feature names to match training data
    input_df = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]], 
        columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    )
    
    # 2. Scale the input using the already fitted scaler
    scaled_features = scaler.transform(input_df)
    
    # 3. Predict the species
    prediction = model.predict(scaled_features)
    
    # prediction is an array with 1 item, so we return the first element
    return prediction[0]
