import pickle
import pandas as pd
def csv_to_binary(csv_file_path, binary_file_path):
    # Read the CSV data
    data = pd.read_csv(csv_file_path, header=None)
    # print(data)
    # Write the data to a binary file
    with open(binary_file_path, 'wb') as f:
        pickle.dump(data, f)
