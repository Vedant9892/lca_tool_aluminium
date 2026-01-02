import os
import pandas as pd
import numpy as np

def load_csv_safely(file_path, expected_columns=None):
    """Load CSV file with error handling and column validation."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        if expected_columns and not all(col in df.columns for col in expected_columns):
            missing_cols = [col for col in expected_columns if col not in df.columns]
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        return df
    except Exception as e:
        raise Exception(f"Error loading CSV file {os.path.basename(file_path)}: {str(e)}")

def get_user_inputs():
    """Get and validate user inputs for aluminum sheet specifications."""
    while True:
        try:
            print("\nAluminum Sheet Calculator")
            print("-" * 30)
            
            # Get dimensions with validation
            while True:
                try:
                    thickness = float(input("Enter sheet thickness (mm): ").strip())
                    if thickness <= 0:
                        raise ValueError("Thickness must be positive")
                    if thickness > 500:  # Reasonable maximum thickness
                        raise ValueError("Thickness seems too large (max 500mm)")
                    break
                except ValueError as e:
                    print(f"Error: {str(e)}")
                    print("Please enter a valid thickness.")
            
            while True:
                try:
                    length = float(input("Enter sheet length (mm): ").strip())
                    if length <= 0:
                        raise ValueError("Length must be positive")
                    if length > 10000:  # Reasonable maximum length
                        raise ValueError("Length seems too large (max 10000mm)")
                    break
                except ValueError as e:
                    print(f"Error: {str(e)}")
                    print("Please enter a valid length.")
            
            while True:
                try:
                    width = float(input("Enter sheet width (mm): ").strip())
                    if width <= 0:
                        raise ValueError("Width must be positive")
                    if width > 5000:  # Reasonable maximum width
                        raise ValueError("Width seems too large (max 5000mm)")
                    break
                except ValueError as e:
                    print(f"Error: {str(e)}")
                    print("Please enter a valid width.")
            
            while True:
                try:
                    num_sheets = int(input("Enter number of sheets: ").strip())
                    if num_sheets <= 0:
                        raise ValueError("Number of sheets must be positive")
                    if num_sheets > 1000:  # Reasonable maximum
                        raise ValueError("Number of sheets seems too large (max 1000)")
                    break
                except ValueError as e:
                    print(f"Error: {str(e)}")
                    print("Please enter a valid number of sheets.")
            
            # Get production route
            print("\nProduction Routes:")
            print("1. Conventional (from bauxite)")
            print("2. Recycled (from scrap)")
            while True:
                route = input("Choose production route (1/2): ").strip()
                if route in ['1', '2']:
                    route = "conventional" if route == '1' else "recycled"
                    break
                print("Error: Please enter 1 or 2")
            
            # Get ore grade for conventional route
            ore_grade = None
            if route == "conventional":
                print("\nOre Grades:")
                print("1. High (>50% Al2O3)")
                print("2. Medium (30-50% Al2O3)")
                print("3. Low (<30% Al2O3)")
                while True:
                    grade_choice = input("Choose ore grade (1/2/3): ").strip()
                    if grade_choice in ['1', '2', '3']:
                        ore_grade = "high" if grade_choice == '1' else "medium" if grade_choice == '2' else "low"
                        break
                    print("Error: Please enter 1, 2, or 3")
            
            # Get energy route
            print("\nEnergy Routes:")
            print("1. Renewable (Solar, Wind, Hydro)")
            print("2. Non-Renewable (Coal, Gas)")
            while True:
                energy = input("Choose energy route (1/2): ").strip()
                if energy in ['1', '2']:
                    energy_route = "renewable" if energy == '1' else "non_renewable"
                    break
                print("Error: Please enter 1 or 2")
            
            return {
                'thickness': thickness,
                'length': length,
                'width': width,
                'num_sheets': num_sheets,
                'route': route,
                'ore_grade': ore_grade,
                'energy_route': energy_route
            }
            
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            return None
        except Exception as e:
            print(f"\nUnexpected error: {str(e)}")
            print("Please try again.")
            if input("\nPress Enter to continue or 'q' to quit: ").lower() == 'q':
                return None

def calculate_weight(thickness, length, width, num_sheets):
    """Calculate total weight of aluminum sheets."""
    try:
        # Density of aluminum (g/cm³)
        density = 2.7
        
        # Convert dimensions to cm
        thickness_cm = thickness / 10
        length_cm = length / 10
        width_cm = width / 10
        
        # Calculate volume in cm³
        volume = thickness_cm * length_cm * width_cm
        
        # Calculate weight in kg
        weight_per_sheet = (volume * density) / 1000
        total_weight = weight_per_sheet * num_sheets
        
        return total_weight
    except Exception as e:
        raise Exception(f"Error calculating weight: {str(e)}")

def get_production_data(weight, route, ore_grade=None, energy_route=None):
    """Get production data based on route, grade, and energy type."""
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        if route == "recycled":
            file_path = os.path.join(base_path, "recycled_route_scrap_1kg_aluminium_100_samples.csv")
            df = load_csv_safely(file_path)
            
            # Get mean values for 1kg production
            metrics = {
                'electricity': df['Total__Electricity_kWh'].mean(),
                'carbon': df['Total__Carbon_kgCO2e'].mean(),
                'natural_gas': df['Total__NaturalGas_Nm3'].mean(),
                'wastewater': df['Total__WasteWater_L'].mean(),
                'quality_score': 0.7  # Base quality score for recycled
            }
            
            # Calculate base cost (example values)
            base_cost = 2.5  # USD per kg for recycled
            
        else:  # conventional route
            if ore_grade.lower() == "high":
                file_path = os.path.join(base_path, "high_grade_bauxite_1kg_samples.csv")
                quality_score = 0.9
                base_cost = 3.0  # USD per kg for high grade
            elif ore_grade.lower() == "medium":
                file_path = os.path.join(base_path, "medium_grade_bauxite_1kg_samples.csv")
                quality_score = 0.75
                base_cost = 2.8  # USD per kg for medium grade
            else:  # low grade
                file_path = os.path.join(base_path, "low_grade_bauxite_1kg_samples.csv")
                quality_score = 0.6
                base_cost = 2.6  # USD per kg for low grade
            
            df = load_csv_safely(file_path)
            
            # Get mean values for 1kg production
            metrics = {
                'electricity': df['Total__Electricity_kWh'].mean(),
                'carbon': df['Total__Carbon_kgCO2e'].mean(),
                'natural_gas': df['Total__NaturalGas_Nm3'].mean(),
                'wastewater': df['Total__WasteWater_L'].mean(),
                'quality_score': quality_score
            }
        
        # Scale metrics by weight
        for key in metrics:
            if key != 'quality_score':
                metrics[key] *= weight
        
        # Adjust metrics based on energy route
        if energy_route == "renewable":
            metrics['carbon'] *= 0.3  # 70% reduction in carbon emissions
            metrics['electricity'] *= 1.1  # 10% more electricity consumption
            base_cost *= 1.15  # 15% premium for renewable energy
        
        # Calculate manufacturing cost components
        carbon_cost = metrics['carbon'] * 0.05  # USD 0.05 per kg CO2
        energy_cost = metrics['electricity'] * 0.1  # USD 0.1 per kWh
        material_cost = base_cost * weight
        transport_cost = weight * 0.2  # USD 0.2 per kg for transport
        
        # Add cost components to metrics
        metrics['carbon_cost'] = carbon_cost
        metrics['energy_cost'] = energy_cost
        metrics['material_cost'] = material_cost
        metrics['transport_cost'] = transport_cost
        metrics['manufacturing_cost'] = material_cost + carbon_cost + energy_cost + transport_cost
        
        return metrics
        
    except Exception as e:
        raise Exception(f"Error calculating production data: {str(e)}")

def main():
    """Main function to run the aluminum sheet calculator."""
    try:
        # Get user inputs
        inputs = get_user_inputs()
        if not inputs:
            return
        
        # Calculate total weight
        total_weight = calculate_weight(
            inputs['thickness'],
            inputs['length'],
            inputs['width'],
            inputs['num_sheets']
        )
        
        # Get production metrics
        metrics = get_production_data(
            total_weight,
            inputs['route'],
            inputs['ore_grade'],
            inputs['energy_route']
        )
        
        # Display results
        print("\nProduction Analysis Results")
        print("=" * 50)
        
        print(f"\nSheet Specifications:")
        print(f"- Dimensions: {inputs['length']:.1f}mm × {inputs['width']:.1f}mm × {inputs['thickness']:.1f}mm")
        print(f"- Number of sheets: {inputs['num_sheets']}")
        print(f"- Total weight: {total_weight:.2f} kg")
        
        print(f"\nProduction Route: {inputs['route'].title()}")
        if inputs['route'] == "conventional":
            print(f"- Ore Grade: {inputs['ore_grade'].title()}")
        print(f"- Energy Type: {inputs['energy_route'].replace('_', ' ').title()}")
        
        print("\nProcess Steps:")
        if inputs['route'] == "conventional":
            print("1. Bauxite mining and crushing")
            print("2. Bayer process (alumina extraction)")
            print("3. Hall-Héroult process (electrolysis)")
        else:
            print("1. Scrap collection and sorting")
            print("2. Scrap pretreatment")
            print("3. Melting and refining")
        print("4. Casting and rolling")
        print("5. Heat treatment")
        print("6. Surface finishing")
        
        print("\nCost Breakdown:")
        print(f"- Material Cost: ${metrics['material_cost']:.2f}")
        print(f"- Energy Cost: ${metrics['energy_cost']:.2f}")
        print(f"- Carbon Cost: ${metrics['carbon_cost']:.2f}")
        print(f"- Transport Cost: ${metrics['transport_cost']:.2f}")
        print(f"- Total Manufacturing Cost: ${metrics['manufacturing_cost']:.2f}")
        
        print("\nResource Usage:")
        print(f"- Electricity: {metrics['electricity']:.2f} kWh")
        print(f"- Carbon Emissions: {metrics['carbon']:.2f} kgCO2e")
        print(f"- Natural Gas: {metrics['natural_gas']:.2f} Nm³")
        print(f"- Wastewater: {metrics['wastewater']:.2f} L")
        
        print("\nQuality Assessment:")
        print(f"- Quality Score: {metrics['quality_score']:.2f} (0.0-1.0 scale)")
        print("Quality Scale Reference:")
        print("  0.9-1.0: Premium Grade (Aerospace, Medical)")
        print("  0.7-0.8: Industrial Grade (Construction, Automotive)")
        print("  0.5-0.6: Commercial Grade (General Purpose)")
        
        print("\nEnvironmental Impact:")
        if inputs['energy_route'] == "renewable":
            print("✓ Using renewable energy reduces carbon emissions by 70%")
            print("✓ Higher initial cost offset by environmental benefits")
        else:
            print("! Consider switching to renewable energy to reduce emissions")
            print("! Carbon costs may increase with future regulations")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please try again with valid inputs.")

if __name__ == "__main__":
    main()