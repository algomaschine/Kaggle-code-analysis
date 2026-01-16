import pandas as pd



# Define a function to check the conditions
def filter_rows(x):
    # Trim whitespace and check if length is less than 4
    if len(x.strip()) < 4:
        return False
    # Check if '------' is in the string
    if '------' in x:
        return False
    
    return True

# Apply the filter function to the desired column (assuming 'ref' as an example)



def process_and_save_csv_to_text(csv_path, output_txt_path):
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Extract the first part of each 'ref' field, assuming it's separated by spaces
    first_elements = df['ref'].str.split().str[0]
    df = df[df['ref'].apply(filter_rows)]
    print(df)

    # Save these first elements to a text file
    urls = set([])
    for e in list(df['ref']):
        if ' ' in e:    e =e.split(' ')[0]
        urls.add(f'kaggle kernels pull {e}')

    with open("getem.bat", 'w') as file:
        for item in urls:
            file.write(item + '\n')
            file.write('sleep 1\n')

# Specify the path to your CSV file and the output text file
csv_path = 'Kaggle_Classification_Kernels.csv'
output_txt_path = 'code.txt'

# Run the function
process_and_save_csv_to_text(csv_path, output_txt_path)
