#pip install shap

from data_org import df

import pandas as pd
import numpy as np
import random
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Assuming you have a DataFrame named 'df' with features and the diagnosis column
X = df.drop(['diagnosis', 'sample_id', 'serialNumber'], axis=1)
y = df['diagnosis']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a scaler
scaler = StandardScaler()

# Fit on the training data
X_train_scaled = scaler.fit_transform(X_train)

# Apply the scaling to the test data
X_test_scaled = scaler.transform(X_test)

# Create a Neural Network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Calculate mean squared error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Use SHAP for feature importance
explainer = shap.DeepExplainer(model, X_train_scaled)
shap_values = explainer.shap_values(X_test_scaled)

# Calculate the mean absolute SHAP values for each feature
mean_abs_shap_values = np.mean(np.abs(shap_values[0]), axis=0)

# Create a DataFrame of the features and their mean absolute SHAP values
shap_df = pd.DataFrame(list(zip(X.columns, mean_abs_shap_values)), columns=['feature', 'mean_abs_shap_val'])

# Sort the DataFrame by the mean absolute SHAP values in descending order and select the top 20 features
top_20_features = shap_df.sort_values('mean_abs_shap_val', ascending=False).head(20)

print(top_20_features)


# Assuming top_20_features is your DataFrame
top_20_features['frequency'] = top_20_features['feature'].str.extract('(\d+)').astype(int)  # Extract frequency from feature

# Create separate DataFrames for Magnitude and Phase
top_20_features_magnitude = top_20_features[top_20_features['feature'].str.contains('Magnitude')].copy()
top_20_features_phase = top_20_features[top_20_features['feature'].str.contains('Phase')].copy()

# Find frequencies that have missing Phase or Magnitude
missing_phase_frequencies = set(top_20_features_magnitude['frequency']) - set(top_20_features_phase['frequency'])
missing_magnitude_frequencies = set(top_20_features_phase['frequency']) - set(top_20_features_magnitude['frequency'])

# Create new Phase and Magnitude entries for these frequencies
new_phase_entries = pd.DataFrame({
    'feature': 'Phase_' + pd.Series(list(missing_phase_frequencies)).astype(str) + '.1',
    'mean_abs_shap_val': np.nan,  # Replace with synthetic data as needed
    'frequency': list(missing_phase_frequencies)
})

new_magnitude_entries = pd.DataFrame({
    'feature': 'Magnitude_' + pd.Series(list(missing_magnitude_frequencies)).astype(str),
    'mean_abs_shap_val': np.nan,  # Replace with synthetic data as needed
    'frequency': list(missing_magnitude_frequencies)
})

# Append the new entries to the original DataFrame
top_20_features_final = pd.concat([top_20_features, new_phase_entries, new_magnitude_entries], ignore_index=True)

# Reorder the rows to make Magnitude and Phase for each frequency adjacent
top_20_features_final.sort_values('frequency', inplace=True)
top_20_features_final.reset_index(drop=True, inplace=True)

features_array = top_20_features_final['feature'].values

# Convert combined_array to a NumPy array if it's not already
combined_array = np.array(features_array)

# Create a list of additional column headers
additional_columns = ['sample_id', 'serialNumber', 'diagnosis']

# Concatenate the two arrays
selected_columns = df[np.concatenate((features_array, additional_columns))]
selected_columns