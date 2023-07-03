from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model from the file you saved it to
model = joblib.load('models/fraud_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)

    # Convert the incoming json to pandas DataFrame
    incoming_data = pd.DataFrame(data)

    # Make prediction using model loaded from disk
    prediction = model.predict(incoming_data)

    # Take the first value of prediction
    output = prediction[0]

    # Return the prediction
    return jsonify(int(output))

if __name__ == '__main__':
    try:
        app.run(port=5000, debug=True)
    except:
        print("Server is exited unexpectedly. Please check the server.")
