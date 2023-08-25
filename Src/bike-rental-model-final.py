import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load and preprocess the dataset
csv_path = "../data/daily-bike-share.csv"
df = pd.read_csv(csv_path)
X = df.drop(columns=["instant", "dteday", "rentals"])
categorical_columns = [
    "season",
    "yr",
    "mnth",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
]
X[categorical_columns] = X[categorical_columns].astype("category")
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
y = df["rentals"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the Random Forest model
rf_model = RandomForestRegressor(
    max_depth=20,
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=100,
    random_state=42,
)
rf_model.fit(X_train, y_train)

# Generate predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Create scatter plot comparing actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Rentals")
plt.ylabel("Predicted Rentals")
plt.title("Actual vs. Predicted Rentals")
plt.show()

# Create a histogram of the residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, edgecolor="k")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.show()

# Create a line plot of actual values and predictions over time
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label="Actual Rentals", marker="o")
plt.plot(y_pred, label="Predicted Rentals", linestyle="dashed", marker="x")
plt.xlabel("Time")
plt.ylabel("Rentals")
plt.title("Actual vs. Predicted Rentals Over Time")
plt.legend()
plt.tight_layout()
plt.show()

# Save the trained model to a file
model_filename = "../models/random_forest_model.joblib"
joblib.dump(rf_model, model_filename)

# In the production environment:
# Load the trained model from the file
loaded_model = joblib.load(model_filename)

# Example new data for prediction
new_data = X.iloc[0:1]  # Use any new data you have

# Use the loaded model to make predictions with feature names
predicted_rentals = loaded_model.predict(new_data)
print("Predicted Rentals:", predicted_rentals)
