import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from feature_select import selected_columns

# Assuming you have a DataFrame named 'selected_columns' with features and the diagnosis column
X = selected_columns.drop(['diagnosis', 'sample_id', 'serialNumber'], axis=1)
y = selected_columns['diagnosis']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)  # Set the number of estimators as desired

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate mean squared error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate R-squared (R²)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Initialize lists to store the results
results = []
random_numbers = []

# Loop through the test set and make predictions
for _ in range(len(X_test)):
    random_number = random.randint(0, len(X_test) - 1)
    random_numbers.append(random_number)

    # Choose a specific input data point to observe the response
    input_data = X_test.iloc[random_number]  # Select the desired input data point

    # Make a prediction for the chosen input data point
    predicted_output = model.predict([input_data])

    # Retrieve the corresponding actual output
    actual_output = y_test.iloc[random_number]  # Select the actual output for the chosen input data point

    # Append the results to the list
    results.append([random_number, input_data, predicted_output, actual_output])

# Create a DataFrame from the results
columns = ['rand numb', 'Input Data', 'Predicted Output', 'Actual Output']
selected_columns_results = pd.DataFrame(results, columns=columns)

# Calculate the difference between predicted output and actual output
selected_columns_results['Difference'] = selected_columns_results['Predicted Output'] - selected_columns_results['Actual Output']

# Print the DataFrame
print(selected_columns_results)
print(selected_columns_results['Difference'])
