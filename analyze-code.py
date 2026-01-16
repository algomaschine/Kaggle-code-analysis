import os
import subprocess
import shutil
from multiprocessing import Pool

def convert_and_move(file_info):
    directory, file, py_dir = file_info
    if file.endswith(".ipynb"):
        # Define the output Python file name
        py_file = file.replace('.ipynb', '.py')
        # Construct the command
        command = f"ipynb-py-convert {os.path.join(directory, file)} {os.path.join(directory, py_file)}"
        
        try:
            # Run the command
            subprocess.run(command, check=True, shell=True)
            print(f"Converted {file} to {py_file}")

            # Move the .py file to the 'py' directory
            shutil.move(os.path.join(directory, py_file), os.path.join(py_dir, py_file))
            print(f"Moved {py_file} to {py_dir}")

        except subprocess.CalledProcessError as e:
            print(f"Failed to convert {file}: {e}")

def main(directory):
    # Path for the 'py' directory
    py_dir = os.path.join(directory, 'py')
    
    # Create the 'py' directory if it does not exist
    if not os.path.exists(py_dir):
        os.makedirs(py_dir)
        print(f"Directory 'py' created at {py_dir}")

    # List all files and prepare file information for multiprocessing
    files = os.listdir(directory)
    file_info = [(directory, file, py_dir) for file in files if file.endswith('.ipynb')]

    # Use a pool of workers to process files in parallel
    with Pool(processes=os.cpu_count()) as pool:
        pool.map(convert_and_move, file_info)

# Specify the directory containing the .ipynb files
directory = '.'

# Run the main function
if __name__ == '__main__':
    main(directory)
