import pandas as pd

# Load the CSV file
df1 = pd.read_csv('train_data (3).csv')

# Get the column names as a list
col_names = df1.columns.tolist()

# Rename the magnitude columns
col_names[3:1003] = ['Magnitude_' + str(col) for col in col_names[3:1003]]

# Rename the phase columns
col_names[1003:] = ['Phase_' + str(col) for col in col_names[1003:]]

# Assign the new column names to the DataFrame
df1.columns = col_names

# Mapping
mapping = {'Positiva': 100, 'midway': 50, 'Negativa': 0}

# Apply the mapping to the 'diagnosis' column
df1['diagnosis'] = df1['diagnosis'].map(mapping)

# Filter the DataFrame and create a new copy
df = df1[df1['serialNumber'] != '2C7707BD9710'].copy()

# Convert 'diagnosis' to numeric, coercing non-numeric values to NaN
df.loc[:, 'diagnosis'] = pd.to_numeric(df['diagnosis'], errors='coerce')

# Replace NaN values with zero
df.loc[df['diagnosis'].isnull(), 'diagnosis'] = 0

# Print the filtered DataFrame
print(len(df))
