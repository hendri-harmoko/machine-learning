import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset
file_path = "/Users/hendriharmoko/Desktop/myLLMs/machine-learning/daily-bike-share.csv"
data = pd.read_csv(file_path)

# Convert 'dteday' to datetime
data['dteday'] = pd.to_datetime(data['dteday'])

# Extract year, month, and day from 'dteday'
data['year'] = data['dteday'].dt.year
data['month'] = data['dteday'].dt.month
data['day'] = data['dteday'].dt.day

# Drop the original 'dteday' column
data.drop(columns=['dteday'], inplace=True)

# List of columns to convert to dummies
categorical_columns = ['season', 'year', 'month', 'holiday', 'weekday', 'workingday', 'weathersit']

# Convert categorical columns to dummy variables
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Select features and target variable
features = ['temp', 'atemp', 'hum', 'windspeed'] + list(data.columns[1:])  # Exclude the 'instant' column
target = 'rentals'

X = data[features]
y = data[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate R^2 score
r2 = r2_score(y_test, y_pred)

print("R^2 Score:", r2)

import matplotlib.pyplot as plt
import numpy as np

# ... (previous code for preprocessing and model fitting)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([0, np.max(y_test)], [0, np.max(y_test)], color='black', linestyle='--')
plt.title('Actual vs. Predicted Bike Rentals')
plt.xlabel('Actual Rentals')
plt.ylabel('Predicted Rentals')
plt.show()