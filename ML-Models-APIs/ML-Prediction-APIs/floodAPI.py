from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)


model = joblib.load("models/decision_tree_model.joblib")
subdiv_encoder = joblib.load("models/subdiv_encoder.joblib")

def preprocess_input(data):
    """Preprocess the input data by encoding and normalizing."""
  
    if data["SUBDIVISIONS"] in subdiv_encoder.classes_:
        data["subdivision_encoded"] = subdiv_encoder.transform([data["SUBDIVISIONS"]])[0]
    else:
        return {"error": "Unknown subdivision"}, 400

  
    columns = ['subdivision_encoded', 'YEAR', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
               'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'ANNUAL']
    try:
        processed_data = {col: data[col] for col in columns}
        return pd.DataFrame([processed_data])
    except KeyError as e:
        return {"error": f"Missing field: {str(e)}"}, 400

def postprocess_output(prediction):
    """Convert numerical prediction to human-readable output."""
    return "YES" if prediction == 1 else "NO"

@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint for flood prediction."""
    try:
        user_data = request.get_json()
        if not user_data:
            return jsonify({"error": "No input data provided"}), 400

        processed_data = preprocess_input(user_data)
        if isinstance(processed_data, tuple):  
            return jsonify(processed_data[0]), processed_data[1]


        prediction = model.predict(processed_data)[0]

       
        result = postprocess_output(prediction)

        return jsonify({
            "SUBDIVISIONS": user_data["SUBDIVISIONS"],
            "YEAR": user_data["YEAR"],
            "PREDICTION": result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
