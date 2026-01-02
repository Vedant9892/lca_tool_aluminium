import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

# Get the current directory path
current_dir = os.path.dirname(os.path.abspath(__file__))

# --- 1. Load Data ---
try:
    # Load dataset with absolute path
    csv_file = os.path.join(current_dir, 'recycled_route_scrap_1kg_aluminium_100_samples.csv')
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found at: {csv_file}")
        
    df = pd.read_csv(csv_file)
    print(f"Successfully loaded data from: {csv_file}")
    print(f"Dataset shape: {df.shape}")

except FileNotFoundError as e:
    print(f"Error: Could not find CSV file: {csv_file}")
    print("Please ensure the file exists in the directory.")
    raise e

except Exception as e:
    print(f"An unexpected error occurred: {str(e)}")
    raise e

# Define feature and target columns
feature_column = 'Scrap_Input_kg'
target_columns = [
    'Scrap_Input_kg',  # Corresponds to 'scrap required'
    'Total__Electricity_kWh',  # Corresponds to 'electricity required'
    'Total__Carbon_kgCO2e',  # Corresponds to 'total carbon emission'
    'Total__NaturalGas_Nm3',
    'Total__WasteWater_L',
    'Total__LiquidFuel_L',
    'Total__Fluorine_g',
    'Total__SO2_g'
]

# Prepare data for training
X = df[[feature_column]].values.reshape(-1, 1)
y = df[target_columns].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
print("\nTraining the model...")
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model performance
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


def predict_factors(required_aluminium: float) -> dict:

    try:
        # Input validation
        if required_aluminium <= 0:
            raise ValueError("Required aluminium must be positive")

        # Create input array for prediction
        input_for_prediction = np.array([[required_aluminium]])

        # Make prediction
        predicted_values_array = model.predict(input_for_prediction)

        # Store predictions in dictionary with proper units
        predicted_values = {}
        units = {
            'Scrap_Input_kg': 'kg',
            'Total__Electricity_kWh': 'kWh',
            'Total__Carbon_kgCO2e': 'kgCO2e',
            'Total__NaturalGas_Nm3': 'Nm³',
            'Total__WasteWater_L': 'L',
            'Total__LiquidFuel_L': 'L',
            'Total__Fluorine_g': 'g',
            'Total__SO2_g': 'g'
        }

        for i, col in enumerate(target_columns):
            value = predicted_values_array[0][i]
            predicted_values[col] = {
                'value': value,
                'unit': units[col]
            }

        return predicted_values

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    try:
        example_required_aluminium = int(input("Enter the required amount of aluminium (in kg): "))
        predicted_results = predict_factors(example_required_aluminium)
        
        print(f"\nPredicted factors for {example_required_aluminium} kg input:")
        print("-" * 50)
        for key, data in predicted_results.items():
            display_name = key.replace('Total__', '').replace('_', ' ')
            print(f"{display_name:20}: {data['value']:10.4f} {data['unit']}")

    except Exception as e:
        print(f"Error in example prediction: {str(e)}")