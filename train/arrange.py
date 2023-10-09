import pandas as pd
import numpy as np

def convert_tensor_to_float(value):
    if isinstance(value, str) and value.startswith('tensor(') and value.endswith(')'):
        try:
            numeric_value = float(value.replace('tensor(', '').replace(')', ''))
            return np.float32(numeric_value)
        except ValueError:
            return np.nan
    else:
        return np.nan

# Replace 'your_file.csv' with the actual path to your CSV file
csv_file = 'datapoints.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Shift values from columns 0, 1, 2, etc., to the Keypoint columns
for i in range(0, 34):
    df[f'Keypoint_{i}'] = df[str(i)]  # Use column label as a string

# Remove the original columns
for i in range(0, 34):
    df.drop(columns=str(i), inplace=True)  # Use column label as a string

# Now 'df' contains the shifted data
# The values from columns 0, 1, 2, etc., are moved to the corresponding Keypoint columns
sorted_df = df.sort_values(by=['ID', 'Frame'])

# Apply the conversion function to the specified columns
columns_to_convert = ['X', 'Y', 'Width', 'Height']
sorted_df[columns_to_convert] = sorted_df[columns_to_convert].applymap(convert_tensor_to_float).astype('float32')

print(sorted_df)
sorted_df.to_csv('datapoints2.csv', index=False)
