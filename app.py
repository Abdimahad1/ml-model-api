from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Create Flask app
app = Flask(__name__)
CORS(app)

# Correct path for model (works on Render or local)
MODEL_DIR = os.path.join(os.getcwd(), 'model_files')
os.makedirs(MODEL_DIR, exist_ok=True)

# Load model artifacts
try:
    model = joblib.load(os.path.join(MODEL_DIR, 'xgboost_model.pkl'))
    label_encoders = joblib.load(os.path.join(MODEL_DIR, 'label_encoders.pkl'))
    feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
    print("✅ Model artifacts loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model artifacts: {str(e)}")
    raise e

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not request.is_json:
            print("❌ Request is not JSON")
            return jsonify({"error": "Request must be JSON", "status": "failed"}), 400

        data = request.get_json()
        print("📥 Received data:", data)

        required_fields = [
            'market', 'founded_year', 'funding_total_usd', 'funding_rounds',
            'country_code', 'city'
        ]
        for field in required_fields:
            if field not in data:
                print(f"❌ Missing field: {field}")
                return jsonify({"error": f"Missing required field: {field}", "status": "failed"}), 400

        funding_fields = [
            'seed', 'venture', 'angel', 'debt_financing',
            'convertible_note', 'equity_crowdfunding',
            'private_equity', 'post_ipo_equity'
        ]
        if not any(data.get(field, 0) > 0 for field in funding_fields):
            print("❌ No valid funding fields provided.")
            return jsonify({
                "error": "No funding types were selected. Please provide at least one source of funding.",
                "status": "failed"
            }), 400

        current_year = datetime.now().year
        business_age = current_year - int(data['founded_year'])

        if business_age < 5:
            print("❌ Business too young:", business_age)
            return jsonify({
                "error": "Business must be at least 5 years old to be considered safe",
                "status": "failed"
            }), 400

        input_data = {
            'name': data.get('name', 'Unknown'),
            'market': data['market'],
            'founded_year': int(data['founded_year']),
            'funding_total_usd': float(data['funding_total_usd']),
            'funding_rounds': int(data['funding_rounds']),
            'seed': float(data.get('seed', 0)),
            'venture': float(data.get('venture', 0)),
            'angel': float(data.get('angel', 0)),
            'debt_financing': float(data.get('debt_financing', 0)),
            'convertible_note': float(data.get('convertible_note', 0)),
            'equity_crowdfunding': float(data.get('equity_crowdfunding', 0)),
            'private_equity': float(data.get('private_equity', 0)),
            'post_ipo_equity': float(data.get('post_ipo_equity', 0)),
            'country_code': data['country_code'],
            'city': data['city'],
            'first_funding_year': int(data.get('first_funding_year', data['founded_year']))
        }

        print("🧪 Cleaned input data for model:", input_data)

        df = pd.DataFrame([input_data])

        # Encode categorical fields
        for col in label_encoders:
            if col in df.columns:
                try:
                    df[col] = label_encoders[col].transform(df[col].astype(str))
                except ValueError as ve:
                    print(f"⚠️ Label encoding failed for column: {col} — {ve}")
                    df[col] = 0

        df = df[feature_names]
        print("📊 Final DataFrame for prediction:\n", df)

        # Model prediction
        probability = model.predict_proba(df)[0][1]
        prediction = model.predict(df)[0]

        # Explanation
        explanation = []
        if business_age >= 10:
            explanation.append("The business has been operating for over 10 years, indicating strong stability.")
        elif business_age >= 5:
            explanation.append("The business has a healthy age of over 5 years, suggesting resilience.")

        if data.get("debt_financing", 0) > 0:
            explanation.append("Debt financing detected, which often reflects financial trust by lenders.")
        if data.get("seed", 0) > 0:
            explanation.append("Seed funding is present, indicating investor confidence at an early stage.")
        if data.get("venture", 0) > 0:
            explanation.append("Venture funding implies high-growth potential and backing.")
        if data.get("equity_crowdfunding", 0) > 0:
            explanation.append("Equity crowdfunding reflects public interest and community support.")

        if not explanation:
            explanation.append("Low funding and minimal history may increase investment risk.")

        print("✅ Prediction:", "Safe" if prediction == 1 else "Not Safe", "| Probability:", probability)

        return jsonify({
            "prediction": "Safe" if prediction == 1 else "Not Safe",
            "probability": round(float(probability), 4),
            "status": "success",
            "explanation": explanation
        })

    except ValueError as ve:
        print("❌ ValueError:", ve)
        return jsonify({"error": f"Invalid input: {str(ve)}", "status": "failed"}), 400
    except Exception as e:
        print("❌ Exception:", e)
        return jsonify({"error": str(e), "status": "failed"}), 500

# Health check
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "required_features": feature_names
    })

# Run app
if __name__ == '__main__':
    # Use PORT from Render environment if available
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port, debug=False)
