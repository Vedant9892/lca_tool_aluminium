import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# --- 1. Load Data ---
try:
    # Load datasets with proper paths
    df3 = pd.read_csv(os.path.join(current_dir, 'high_grade_bauxite_1kg_samples.csv'))
    df2 = pd.read_csv(os.path.join(current_dir, 'medium_grade_bauxite_1kg_samples.csv'))
    df1 = pd.read_csv(os.path.join(current_dir, 'low_grade_bauxite_1kg_samples.csv'))

except FileNotFoundError as e:
    print(f"Error: Could not find one or more CSV files in the directory: {current_dir}")
    print("Please ensure the following files exist in the directory:")
    print("- high_grade_bauxite_1kg_samples.csv")
    print("- medium_grade_bauxite_1kg_samples.csv")
    print("- low_grade_bauxite_1kg_samples.csv")
    raise e

except Exception as e:
    print(f"An unexpected error occurred: {str(e)}")
    raise e


# --- 2. Combine and Preprocess Data ---
df_combined = pd.concat([df1, df2, df3], ignore_index=True)
df_combined.drop_duplicates(inplace=True)
df_combined = df_combined.drop('Sample_ID', axis=1)

# Identify 'Total' and individual energy/environment columns
total_columns = [col for col in df_combined.columns if col.startswith('Total__')]
individual_columns = [col for col in df_combined.columns if col not in total_columns and col not in ['Grade']]

# Create X_predict and y_predict for the multi-output regression model
X_predict = df_combined[total_columns]
y_predict = df_combined[individual_columns]

# Calculate median values for each 'Total' energy and environment feature for imputation
medians_for_total_imputation = X_predict.median()

# Calculate the average values for each 'Total' resource for each Grade
average_total_resources_per_grade = df_combined.groupby('Grade')[total_columns].mean()


# --- 3. Split Data for Multi-output Model ---
X_predict_train, X_predict_test, y_predict_train, y_predict_test = train_test_split(
    X_predict, y_predict, test_size=0.2, random_state=42
)

# --- 4. Choose and Train Multi-output Model ---
multi_output_model = RandomForestRegressor(random_state=42)
multi_output_model.fit(X_predict_train, y_predict_train)

# --- 5. Implement Input, Imputation, Prediction, and Output ---

# Prompt the user to enter the amount of aluminum in kilograms
while True:
    try:
        aluminum_amount_input = float(input("Enter the amount of aluminum (in kg): "))
        if aluminum_amount_input <= 0:
            print("Please enter a positive numerical value.")
        else:
            break
    except ValueError:
        print("Invalid input. Please enter a numerical value.")

# Prompt the user to select the grade manually
while True:
    selected_grade = input("Enter the desired grade (High, Medium, Low): ").capitalize()
    if selected_grade in ['High', 'Medium', 'Low']:
        break
    else:
        print("Invalid grade. Please enter 'High', 'Medium', or 'Low'.")

# Calculate the required total resources based on the selected grade and aluminum amount
if selected_grade in average_total_resources_per_grade.index:
    required_total_resources = average_total_resources_per_grade.loc[selected_grade] * aluminum_amount_input

    # Define a function for the final output
    def display_final_output_totals(aluminum_amount, grade, estimated_total_resources_df):
        """Displays the final estimated total energy and environment resources."""
        print("\n" + "="*50)
        print("Final Estimated Total Resource Requirements")
        print("="*50)
        print(f"Aluminum Amount Input: {aluminum_amount} kg")
        print(f"Selected Grade: {grade}")

        print("\nEstimated Total Energy and Environment Resources:")
        print(estimated_total_resources_df.to_frame().T) # Display as transposed for better readability #DISPLAY
        print("="*50)

    # Call the final output function
    display_final_output_totals(aluminum_amount_input, selected_grade, required_total_resources)

else:
    print(f"\nCould not find average total resource data for selected grade: {selected_grade}")
