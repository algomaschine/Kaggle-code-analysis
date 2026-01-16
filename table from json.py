import pandas as pd
import os
import json
from pandas import json_normalize

def load_jsons_to_dataframe(directory):
    dataframes = []  # List to store individual DataFrames created from each JSON file

    # Walk through the directory and process each JSON file
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            
            # Open and load the JSON file
            try:
                with open(file_path, 'r') as file:
                    data_dict = json.load(file)
            except:
                print('error:',file_path)
                continue    
            # Flatten the JSON structure if needed and convert to DataFrame
            # `json_normalize` can be used to flatten nested JSON
            df = json_normalize(data_dict)
            
            # Append the resulting DataFrame to the list
            dataframes.append(df)

    # Concatenate all DataFrames into one
    if dataframes:
        full_df = pd.concat(dataframes, ignore_index=True)
    else:
        full_df = pd.DataFrame()  # Return an empty DataFrame if no JSON files found

    return full_df

# Usage
directory = 'py\\json'
final_df = load_jsons_to_dataframe(directory)
print(final_df)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder
import graphviz
from sklearn import tree

# Assuming `final_df` is your DataFrame loaded from JSON files
#final_df = pd.read_csv("your_data.csv")  # Replace with your DataFrame loading method

# Step 1: Prepare the Data
# Encoding categorical variables if any
label_encoders = {}
for column in final_df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    final_df[column] = le.fit_transform(final_df[column])
    label_encoders[column] = le

# Step 2: Split the Data into features and target
# Assume 'target_column' is the name of your target variable
X = final_df.drop('target_column', axis=1)  # features
y = final_df['target_column']  # target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train a Decision Tree Model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 4: Visualize the Decision Tree
dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=X.columns,  
                                class_names=str(clf.classes_),
                                filled=True, rounded=True,  
                                special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.render("decision_tree")  # Saves the tree as a PDF file

# Optionally display inline (if using Jupyter Notebook)
graph
