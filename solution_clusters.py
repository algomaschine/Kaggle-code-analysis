import os
import json
import pandas as pd
from pandas import json_normalize
import logging

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_and_clean_json(file_path):
    """Reads a JSON file and cleans it if necessary, returning the JSON data."""
    try:
        with open(file_path, 'r') as file:
            file_data = file.read()
        # Find the first { and the last }
        first_brace = file_data.find('{')
        last_brace = file_data.rfind('}') + 1
        if first_brace == -1 or last_brace == 0:
            raise ValueError("Not a valid JSON format")
        clean_data = file_data[first_brace:last_brace]
        data_dict = json.loads(clean_data)
        if file_data[first_brace:last_brace] != file_data:  # Only write if changes were made
            with open(file_path, 'w') as file:
                file.write(clean_data)
        return data_dict
    except Exception as e:
        logging.error(f"Failed to process {file_path}: {e}")
        return None

def load_jsons_from_directories(args):
    dataframes = []
    for directory in args:
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                print(filename)
                file_path = os.path.join(directory, filename)
                data_dict = read_and_clean_json(file_path)
                if data_dict is not None:
                    df = json_normalize(data_dict)
                    dataframes.append(df)
    # Concatenate all DataFrames into one
    if dataframes:
        return pd.concat(dataframes, ignore_index=True)
    return pd.DataFrame()  # Return an empty DataFrame if no JSON files found

# Example of how to call the function
directory1 = '.'
directory2 = '.\\num'
resulting_dataframe = load_jsons_from_directories([directory1, directory2])
resulting_dataframe.to_csv('kaggle_solutions.csv', index=False)
