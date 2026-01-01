import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

# Assume density of aluminum is 2.7 g/cm³ or 2700 kg/m³
ALUMINUM_DENSITY = 2700 # kg/m³

# Global variable to hold the trained model and features/targets for training
trained_model = None
training_features = None
training_targets = None
training_feature_columns = None # Store the list of feature columns used during training

def calculate_volume_and_weight(outer_radius_cm, inner_radius_cm, length_cm):
    """
    Calculates the volume and weight of an aluminum pipe.

    Args:
        outer_radius_cm (float): Outer radius of the pipe in centimeters.
        inner_radius_cm (float): Inner radius of the pipe in centimeters.
        length_cm (float): Length of the pipe in centimeters.

    Returns:
        tuple: A tuple containing:
            - volume_m3 (float): Volume of the pipe in cubic meters.
            - weight_kg (float): Weight of the pipe in kilograms.
    """
    # Convert dimensions to meters
    outer_radius_m = outer_radius_cm / 100
    inner_radius_m = inner_radius_cm / 100
    length_m = length_cm / 100

    # Calculate volume in cubic meters
    volume_m3 = np.pi * (outer_radius_m**2 - inner_radius_m**2) * length_m

    # Calculate weight in kilograms
    weight_kg = volume_m3 * ALUMINUM_DENSITY

    return volume_m3, weight_kg

def load_and_preprocess_data():
    """
    Loads and preprocesses the data for model training.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Load conventional route datasets
    df_conventional_renewable = pd.read_csv(os.path.join(base_path, 'aluminium_pipe_renewable_energy_200.csv'))
    df_conventional_non_renewable = pd.read_csv(os.path.join(base_path, 'aluminium_pipe_non_renewable_energy_200.csv'))
    
    # Combine datasets
    df_combined = pd.concat([df_conventional_renewable, df_conventional_non_renewable])
    
    # Define features and targets
    feature_columns = ['outer_radius_m', 'inner_radius_m', 'length_m', 'volume_m3', 'weight_kg']
    target_columns = ['manufacturing_cost_usd', 'quality_score', 'electricity_kwh', 'carbon_kgCO2e', 
                     'natural_gas_Nm3', 'wastewater_L', 'transport_cost_usd']
    
    # Split into features and targets
    features = df_combined[feature_columns]
    targets = df_combined[target_columns]
    
    return features, targets

def train_model():
    """
    Trains the machine learning model.
    """
    global trained_model, training_features, training_targets, training_feature_columns
    
    try:
        features, targets = load_and_preprocess_data()
        
        if features is None or targets is None:
            print("Could not load or preprocess data for training.")
            return
        
        training_features = features  # Store features and targets for potential future use
        training_targets = targets
        training_feature_columns = features.columns.tolist()  # Store the list of feature columns
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
        
        # Choose a regression model (RandomForestRegressor) and instantiate it
        model = RandomForestRegressor(random_state=42)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Evaluate the model's performance
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
        r2 = r2_score(y_test, y_pred, multioutput='raw_values')
        
        print(f"\nModel training complete. Evaluation metrics:")
        for i, target_col in enumerate(targets.columns):
            print(f"\n{target_col}:")
            print(f"Mean Squared Error: {mse[i]:.2f}")
            print(f"R-squared: {r2[i]:.2f}")
        
        trained_model = model  # Store the trained model globally
        return True
        
    except Exception as e:
        print(f"Error during model training: {e}")
        return False

def train_and_save_model():
    """
    Trains a machine learning model and saves it to a file.
    """
    try:
        # Load and preprocess data
        data = load_and_preprocess_data()
        
        # Define features and target
        features = ['outer_radius_m', 'inner_radius_m', 'length_m', 'volume_m3', 'weight_kg']
        target = 'manufacturing_cost_usd'
        
        X = data[features]
        y = data[target]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate the model
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print(f"Model Performance:")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R² Score: {r2:.2f}")
        
        # Save the model
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, 'aluminum_pipe_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved successfully to: {model_path}")
        
        return model
        
    except Exception as e:
        print(f"Error during model training: {e}")
        return None

def predict_pipe_metrics(outer_radius_cm, inner_radius_cm, length_cm, route, ore_grade=None, energy_route=None):
    """
    Predicts manufacturing metrics for an aluminum pipe using a trained ML model.

    Args:
        outer_radius_cm (float): Outer radius of the pipe in centimeters.
        inner_radius_cm (float): Inner radius of the pipe in centimeters.
        length_cm (float): Length of the pipe in centimeters.
        route (str): The chosen route ('conventional' or 'recycled').
        ore_grade (str, optional): The chosen ore grade ('high', 'medium', or 'low'). Required for conventional route.
        energy_route (str, optional): The chosen energy route ('renewable' or 'non_renewable'). Required for both routes.

    Returns:
        dict: A dictionary containing the predicted metrics or an error message.
    """
    global trained_model, training_feature_columns

    if trained_model is None or training_feature_columns is None:
        return {"error": "Machine learning model not trained or training feature columns not available. Please run the script to train the model first."}

    # Calculate volume and weight
    volume_m3, weight_kg = calculate_volume_and_weight(outer_radius_cm, inner_radius_cm, length_cm)

    # --- Feature Preparation for Model Prediction ---
    # Create a dictionary to hold feature values, initialized with zeros
    input_features_dict = {col: 0 for col in training_feature_columns}

    # Populate the dictionary with calculated weight and user inputs
    if 'weight_kg' in input_features_dict:
        input_features_dict['weight_kg'] = weight_kg
    else:
        print("Warning: 'weight_kg' not found in expected feature columns.")


    # Update one-hot encoded features based on user input
    if f'Route_{route}' in input_features_dict:
         input_features_dict[f'Route_{route}'] = 1
    # Note: Assuming drop_first=True during encoding, so only 'Route_recycled' might exist.
    # If route is 'conventional', Route_recycled will be 0.

    if f'Energy Route_{energy_route}' in input_features_dict:
         input_features_dict[f'Energy Route_{energy_route}'] = 1
    # Note: Assuming drop_first=True, so only 'Energy Route_renewable' might exist.

    if route == 'conventional' and ore_grade is not None:
         if f'ore_grade_{ore_grade}' in input_features_dict:
              input_features_dict[f'ore_grade_{ore_grade}'] = 1
         # Handle the case where ore_grade is 'high' - if 'ore_grade_high' exists as a feature
         elif ore_grade == 'high' and 'ore_grade_high' in input_features_dict:
              input_features_dict['ore_grade_high'] = 1


    # If 'ore_grade_not applicable' exists as a feature column and the route is recycled or conventional with no specified grade, set it to 1.
    if 'ore_grade_not applicable' in input_features_dict and (route == 'recycled' or (route == 'conventional' and ore_grade is None)):
         input_features_dict['ore_grade_not applicable'] = 1


    # Convert the input features dictionary to a DataFrame row with the correct column order
    try:
        features_df = pd.DataFrame([input_features_dict], columns=training_feature_columns)
         # Ensure dtypes match training data if necessary, though RandomForestRegressor is usually flexible.
         # features_df = features_df.astype(training_features.dtypes) # Uncomment if dtype mismatch is an issue

    except ValueError as e:
         return {"error": f"Feature mismatch: Could not create input features DataFrame with expected columns. Details: {e}. Expected columns: {training_feature_columns}"}


    # Make predictions using the trained model
    try:
        predictions = trained_model.predict(features_df)[0] # Get the first row of predictions
    except Exception as e:
        return {"error": f"An error occurred during model prediction: {e}"}

    # --- Assign Predicted Values to Metrics Dictionary ---
    # The order of predictions depends on the order of target columns in your training data (y_train).
    # Assuming the target columns were: 'manufacturing_cost_usd', 'quality_score', 'electricity_kwh', 'carbon_kgCO2e', 'natural_gas_Nm3', 'wastewater_L', 'transport_cost_usd'
    # You need to map the predictions back to the correct metric names.
    # A reliable way is to store the list of target columns when training the model.

    # For now, let's assume the order based on the target_columns list in load_and_preprocess_data.
    # You should verify this against the actual order in your y_train.columns.

    predicted_metrics = {
        "Cost Breakdown": {
            "Material Cost": predictions[0], # Assuming index 0 is manufacturing_cost_usd (needs refinement to separate material cost)
            "Energy Cost": 0.0, # Placeholder - needs to be derived from predicted resource usage and prices
            "Carbon Cost": predictions[3], # Assuming index 3 is carbon_kgCO2e, need to convert to cost if required
            "Transport Cost": predictions[6], # Assuming index 6 is transport_cost_usd
            "Total Manufacturing Cost": predictions[0] # Assuming index 0 is manufacturing_cost_usd
        },
        "Resource Usage": {
            "Electricity kWh": predictions[2], # Assuming index 2 is electricity_kwh
            "Carbon kgCO2e": predictions[3], # Assuming index 3 is carbon_kgCO2e
            "NaturalGas Nm3": predictions[4], # Assuming index 4 is natural_gas_Nm3
            "WasteWater L": predictions[5], # Assuming index 5 is wastewater_L
            "Quality grade": predictions[1] # Assuming index 1 is quality_score
        }
    }
    # Note: The mapping of prediction indices to metrics depends *entirely* on the order of columns in your training targets (y_train).
    # You will likely need to adjust these indices (0 to 6) based on the actual column order used during training.
    # Also, Material Cost and Energy Cost in the Cost Breakdown are currently placeholders or directly using total manufacturing cost/carbon cost.
    # You might need to refine the target variables during training if you want the model to predict these components separately,
    # or calculate them after prediction based on predicted resource usage and cost factors.


    return predicted_metrics

if __name__ == '__main__':
    # Train the model when the script is run
    print("Training the machine learning model...")
    train_model()
    print("-" * 30)

    while True:
        product_choice = input("Would u like to analyze pipe? (y or n): ").lower()
        if product_choice == 'y':
            break
        else:
            print("Invalid product choice. Please select 'y'.")

    print("\nPlease provide the dimensions for the Aluminum Pipe:")
    try:
        outer_radius = float(input("Enter outer radius in cm: "))
        inner_radius = float(input("Enter inner radius in cm: "))
        length = float(input("Enter length in cm: "))
    except ValueError:
        print("Invalid input. Please enter numerical values for dimensions.")
        exit() # Exit if dimensions are not valid numbers

    print("\nProduction Routes:")
    print("1. Conventional (from bauxite)")
    print("2. Recycled (from scrap)")
    while True:
        route_choice = input("Choose production route (1/2): ")
        if route_choice == '1':
            route = 'conventional'
            break
        elif route_choice == '2':
            route = 'recycled'
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

    ore_grade = None
    if route == 'conventional':
        print("\nOre Grades:")
        print("1. High (>50% Al2O3)")
        print("2. Medium (30-50% Al2O3)")
        print("3. Low (<30% Al2O3)")
        while True:
            ore_grade_choice = input("Choose ore grade (1/2/3): ")
            if ore_grade_choice == '1':
                ore_grade = 'high'
                break
            elif ore_grade_choice == '2':
                ore_grade = 'medium'
                break
            elif ore_grade_choice == '3':
                ore_grade = 'low'
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")

    print("\nEnergy Routes:")
    print("1. Renewable (Solar, Wind, Hydro)")
    print("2. Non-Renewable (Coal, Gas)")
    while True:
        energy_route_choice = input("Choose energy route (1/2): ")
        if energy_route_choice == '1':
            energy_route = 'renewable'
            break
        elif energy_route_choice == '2':
            energy_route = 'non_renewable'
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")


    predicted_results = predict_pipe_metrics(outer_radius, inner_radius, length, route, ore_grade, energy_route)

    if "error" in predicted_results:
        print("Error:", predicted_results["error"])
    else:
        print("\n--- Prediction Results ---")
        print("Cost Breakdown:")
        for metric, value in predicted_results["Cost Breakdown"].items():
            print(f"- {metric}: ${value:.2f}") # Assuming cost metrics are in USD

        print("\nResource Usage:")
        for metric, value in predicted_results["Resource Usage"].items():
             if metric == "Quality grade":
                 print(f"- {metric}: {value:.2f}")
             else:
                print(f"- {metric}: {value:.2f}") # Resource usage metrics (e.g., kWh, kgCO2e, Nm3, L)
