import requests

data = {"features": [1, 2, 3, ...]}  # Replace with your input data
response = requests.post("http://localhost:5000/predict", json=data)
prediction = response.json()
print("Predicted Rentals:", prediction["prediction"])
