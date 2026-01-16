import openai
import os
import glob
import time
import json
import re
APIKEY = ' YOUT GPT API KEY'
project_description = """
Please make a detailed step by step summary of this code that includes the following sections (these will be fields in our code markup data storage)
1) if the code has to do with image recognition, then set image_recog = True
2) data cleansing methods used: removing data or imputation (method)
3) data preprocessing methods used, details: normalization, one-hot-encoding, etc
4) feature reduction techniques used
5) new feature generation techiques used
6) data analysis such as clustering 
7) target analysis: defining classes, estimating of frequencies target values for frequencies, balance, etc
8) what kind of task is it: time series, classification, etc
9) type of forecasting used: decimal value, value at time ahead, class probability, binary, other (describe)
10) do we have post-processing of forecasted target once forecasted?
11) were there any other manipulations not mentioned yet? describe.
12) architecture: one model or a chain of models?
13) do we have folds in testing?
14) meta-parameters optimization approach used
15) shuffling train/test or not, percetages used
16) testing on evaluation set (outside train/test), percetages used
17) what algo or bunch of algos used and why
17.1) what framework package or list of packages is used for ML implementation
18) rank complexity from 1 to 10
19) rank quality of solition 1 to 10
20) what are the weaknesses of the solution and how those can be improved?

Generate a json-structured file content that stores data for the fields 1 to 20. It should be neatly formated json, human-readable with tabs, etc.
Below is an example of such json, feel free to add extra elements depending on the complexity of analyzed code.
{
	"image_recog": true,
	"data_cleansing": "None",
	"data_preprocessing": {
		"methods_used": ["Normalization"],
		"details": "Images are normalized using imagenet_stats"
	},
	"feature_reduction": "None",
	"new_feature_generation": "None",
	"data_analysis": "None",
	"target_analysis": {
		"classes_defined": true,
		"target_value_frequencies": "Not estimated",
		"target_balance": "Not analyzed"
	},
	"task_type": "Classification",
	"forecasting_type": "Class probability",
	"post_processing": "None",
	"other_manipulations": "None",
	"architecture": "Single model",
	"fold_testing": false,
	"meta_parameters_optimization": "None",
	"train_test_split": {
		"shuffling": true,
		"train_percentage": 80,
		"test_percentage": 20
	},
	"evaluation_set_testing": {
		"outside_train_test": false,
		"percentage_used": "Not specified"
	},
	"algorithms_used": ["CNN"],
	"framework_package": "FastAI",
	"complexity_rank": 6,
	"quality_rank": 7,
	"weaknesses": "Limited exploration of data analysis and feature engineering. Lack of hyperparameter optimization."
}

"""

def chat_with_gpt(prompt):
    openai.api_key = APIKEY  # Always use secure ways to handle your API key.

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=0.9,
            messages=[
                {"role": "system", "content": "You are a python code analyzer."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content']
    except openai.error.RateLimitError as e:
        error_message = str(e)
        wait_time_str = re.findall(r'\b\d+m?\d+s\b', error_message)
        if wait_time_str:
            wait_time_seconds = convert_to_seconds(wait_time_str[0])
            wait_time_seconds += 10  # Add 10 seconds buffer
            print(f"Rate limit reached, waiting for {wait_time_seconds} seconds.")
            time.sleep(wait_time_seconds)
        return chat_with_gpt(prompt)  # Retry after waiting
    except Exception as ex:
        print(f"An error occurred: {ex}")
        return None

def convert_to_seconds(time_str):
    minutes = re.findall(r'(\d+)m', time_str)
    seconds = re.findall(r'(\d+)s', time_str)
    total_seconds = int(minutes[0]) * 60 if minutes else 0
    total_seconds += int(seconds[0]) if seconds else 0
    return total_seconds


import os
import glob
import time

def json_files_exist(base_directory, base_filename):
    json_directory = os.path.join(base_directory, 'json')
    img_filename = os.path.join(json_directory, f"{base_filename}-IMG.json")
    num_filename = os.path.join(json_directory, f"{base_filename}-NUM.json")
    return os.path.exists(img_filename) or os.path.exists(num_filename)

def process_file(directory, file_path):
    print(f"Processing {file_path}, size: {os.path.getsize(file_path)} bytes")
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    
    if json_files_exist(directory, base_filename):
        print(f"Analysis JSON files for '{file_path}' already exist, skipping API request.")
        return

    with open(file_path, 'r', encoding='utf-8') as file:
        code = file.read() + "\n" + project_description
        response = chat_with_gpt(code)

        if response is None:
            time.sleep(30)
            return

        json_directory = os.path.join(directory, 'json')
        if not os.path.exists(json_directory):
            os.makedirs(json_directory)

        new_file_path = os.path.join(json_directory, f"{base_filename}-IMG.json" if '\"image_recog\": true' in response else f"{base_filename}-NUM.json")

        with open(new_file_path, 'w') as new_file:
            new_file.write(response)
        print(f"Analysis results saved to {new_file_path}")
        time.sleep(5)

def process_files_in_size_order(directory):
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    files = glob.glob(os.path.join(directory, '*.py'))
    files_sorted_by_size = sorted(files, key=os.path.getsize)

    for file_path in files_sorted_by_size:
        process_file(directory, file_path)
        

if __name__ == "__main__":
    directory = 'py'
    process_files_in_size_order(directory)
