import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow frontend requests

# Load the trained Voting Classifier model
with open("E:\\E DRIVE\\CAPSTONE_PROJECT\\voting_classifier.pkl", "rb") as file:
    model = pickle.load(file)

# Load training data
df = pd.read_csv("E:\\E DRIVE\\CAPSTONE_PROJECT\\Train_data.csv\\Train_data.csv")

# Apply Label Encoding on categorical features
service_encoder = LabelEncoder()
flag_encoder = LabelEncoder()
protocol_encoder = LabelEncoder()

df["service"] = service_encoder.fit_transform(df["service"])
df["flag"] = flag_encoder.fit_transform(df["flag"])
df["protocol_type"] = protocol_encoder.fit_transform(df["protocol_type"])

# Define numerical features for StandardScaler
numerical_features = [
    "src_bytes", "dst_bytes", "logged_in",
    "same_srv_rate", "diff_srv_rate", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate"
]

# Fit StandardScaler only on numerical features
scaler = StandardScaler()
scaler.fit(df[numerical_features])  # Fit only on numerical features

@app.route('/submit', methods=['POST'])
def receive_data():
    data = request.json  # Receive JSON data from frontend
    print("Received Data:", data)

    try:
        # Encode categorical features (handle unseen values safely)
        service_encoded = service_encoder.transform([data["service"]])[0] if data["service"] in service_encoder.classes_ else -1
        flag_encoded = flag_encoder.transform([data["flag"]])[0] if data["flag"] in flag_encoder.classes_ else -1
        protocol_encoded = protocol_encoder.transform([data["protocol_type"]])[0] if data["protocol_type"] in protocol_encoder.classes_ else -1

        # Check for invalid categorical encodings
        if -1 in [service_encoded, flag_encoded, protocol_encoded]:
            return jsonify({"error": "Invalid categorical value (service, flag, or protocol_type)"}), 400

        # Extract and convert numerical features
        numerical_values = np.array([
            int(data.get("src_bytes", 0)),
            int(data.get("dst_bytes", 0)),
            int(data.get("logged_in", 0)),
            float(data.get("same_srv_rate", 0.0)),
            float(data.get("diff_srv_rate", 0.0)),
            int(data.get("dst_host_srv_count", 0)),
            float(data.get("dst_host_same_srv_rate", 0.0)),
            float(data.get("dst_host_diff_srv_rate", 0.0)),
        ]).reshape(1
        , -1)

        # Apply StandardScaler on numerical features
        numerical_scaled = scaler.transform(numerical_values)

        # Combine categorical and scaled numerical features
        features = np.concatenate(([service_encoded, flag_encoded], numerical_scaled[0]))

        # Predict using the model
        # Predict using the model
        prediction = model.predict([features])[0]  

        # Determine the label
        label = "Normal" if prediction == 1 else "Malicious" # Normal=1,Malicious=0 

        # Return the prediction response
        response = {
            "message": "Prediction successful",
            "prediction": int(prediction),  # Convert NumPy integer to standard int
            "label": label  # Add label for clarity
        }
        return jsonify(response)


    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
