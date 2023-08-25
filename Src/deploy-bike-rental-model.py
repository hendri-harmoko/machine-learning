from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
loaded_model = joblib.load("../Models/random_forest_model.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()  # Get data from POST request
        # Make predictions using the loaded model
        predicted_rentals = loaded_model.predict([data["features"]])
        return jsonify({"prediction": predicted_rentals.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
