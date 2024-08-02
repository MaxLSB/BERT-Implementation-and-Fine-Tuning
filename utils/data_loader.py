import json

# Load the data from the json file
def load_data(path):
    
    with open(path, 'r') as f:
        dataset = json.load(f)
    return dataset
