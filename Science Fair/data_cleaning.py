import os
import pandas as pd

def combine_csv(folder_path, output_file):
    # List all files in the folder
    all_files = os.listdir(folder_path)

    # Filter for CSV files
    csv_files = [f for f in all_files if f.endswith('.csv')]

    # Container for data from each CSV file
    all_dataframes = []

    for filename in csv_files:
        full_path = os.path.join(folder_path, filename)
        df = pd.read_csv(full_path, index_col=None, header=0)
        all_dataframes.append(df)

    # Concatenate all dataframes
    combined_csv = pd.concat(all_dataframes, axis=0, ignore_index=True)

    # Save the combined CSV to a file
    combined_csv.to_csv(output_file, index=False)
    print(f"Combined CSV created at {output_file}")

# Usage
folder_path = 'DataSets'  # Replace with your folder path
output_file = 'NORADdata.csv'  # Replace with your desired output file name
combine_csv(folder_path, output_file)
