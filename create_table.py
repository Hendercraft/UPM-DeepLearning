import json
import pandas as pd

# Example JSON data (replace this with your actual data loading method)
with open('data/run3/combined.json', 'r') as file:
    json_data = json.load(file)
# Flatten the data and extract relevant information
flattened_data = []

# Function to remove suffixes after the second underscore in keys
def remove_secondary_suffixes(data):
    modified_data = []
    for experiment_group in data:
        new_group = []
        for experiment in experiment_group:
            new_experiment = {}
            for key, value in experiment.items():
                # Find the second underscore and remove it and anything after
                parts = key.split('_')
                if len(parts) > 2 and parts[0] == 'val':
                    new_key = '_'.join(parts[:2])
                    new_experiment[new_key] = value
                else:
                    new_experiment[key] = value
            new_group.append(new_experiment)
        modified_data.append(new_group)
    return modified_data


for experiment_group in remove_secondary_suffixes(json_data):
    for i in range(0, len(experiment_group), 2):
        setup = experiment_group[i]
        results = experiment_group[i + 1]

        # Initialize a dictionary to store the selected metrics
        selected_metrics = {}

        # Iterate over the keys in results and select the appropriate metrics
        for key, value in results.items():
            if key.startswith("val_"):
                # Check if the key has a suffix and select the last epoch's value
                if key[-2] == '_':  # e.g., 'val_loss_1'
                    suffix = key[-1]
                    if setup['epochs'] == int(suffix):
                        new_key = key[:-2]  # Remove the suffix
                        selected_metrics[new_key] = value[-1]
                else:
                    # For metrics without suffix, directly take the last value
                    selected_metrics[key] = value[-1]

        # Add training time to selected_metrics
        selected_metrics['training_time'] = setup['training_time']

        # Combine with the experiment setup information
        combined = {**setup, **selected_metrics}
        flattened_data.append(combined)

# Create a DataFrame
df = pd.DataFrame(flattened_data)
df.to_csv(path_or_buf="./run3.csv")
# Display the DataFrame
print(df)