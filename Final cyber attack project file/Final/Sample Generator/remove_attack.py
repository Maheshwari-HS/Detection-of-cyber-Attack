import pandas as pd
import os

def remove_attack_columns():
    """
    Remove attack_type, attack_category, and timestamp columns from sample_loader.csv
    and save the processed data to Processed_sample folder
    """
    
    # Input file path
    input_file = 'Sample adder/sample_loader.csv'
    
    # Output file path
    output_dir = 'Processed_sample'
    output_file = os.path.join(output_dir, 'sample_loader_processed.csv')
    
    try:
        # Read the CSV file
        print(f"Reading {input_file}...")
        df = pd.read_csv(input_file)
        
        # Display original columns
        print(f"Original columns: {list(df.columns)}")
        print(f"Original shape: {df.shape}")
        
        # Check if attack_type, attack_category, and timestamp columns exist
        columns_to_remove = ['attack_type', 'attack_category', 'timestamp']
        existing_columns_to_remove = [col for col in columns_to_remove if col in df.columns]
        
        if not existing_columns_to_remove:
            print("Warning: No attack_type, attack_category, or timestamp columns found to remove.")
            return
        
        # Remove the specified columns
        print(f"Removing columns: {existing_columns_to_remove}")
        df_processed = df.drop(columns=existing_columns_to_remove)
        
        # Display processed columns
        print(f"Processed columns: {list(df_processed.columns)}")
        print(f"Processed shape: {df_processed.shape}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the processed data
        print(f"Saving processed data to {output_file}...")
        df_processed.to_csv(output_file, index=False)
        
        print(f"Successfully processed and saved to {output_file}")
        print(f"Removed {len(existing_columns_to_remove)} columns: {existing_columns_to_remove}")
        
    except FileNotFoundError:
        print(f"Error: {input_file} not found in the current directory.")
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    remove_attack_columns()
