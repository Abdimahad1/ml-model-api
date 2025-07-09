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

# Model directory
MODEL_DIR = os.path.join(os.getcwd(), 'model_files')
os.makedirs(MODEL_DIR, exist_ok=True)

# Load artifacts
try:
    model = joblib.load(os.path.join(MODEL_DIR, 'xgboost_model.pkl'))
    label_encoders = joblib.load(os.path.join(MODEL_DIR, 'label_encoders.pkl'))
    feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
    print("✅ Model artifacts loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model artifacts: {str(e)}")
    raise e

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not request.is_json:
            print("❌ Invalid: Request not JSON")
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

        has_funding = any(data.get(field, 0) > 0 for field in funding_fields)

        current_year = datetime.now().year
        business_age = current_year - int(data['founded_year'])

        # If business is under 5 years old, still predict but flag risky
        if business_age < 5:
            explanation = [
                f"The business is very young ({business_age} years old), and younger businesses generally carry high risk."
            ]
            prediction = 0
            probability = 0.2
            return jsonify({
                "prediction": "Not Safe",
                "probability": probability,
                "status": "success",
                "explanation": explanation
            })

        # If no funding fields are present, treat as high risk instead of 400
        if not has_funding:
            explanation = [
                "No funding rounds were detected. This appears to be the first round for this business. Proceed with caution and perform due diligence before investing."
            ]
            prediction = 1   # treat as Safe
            probability = 0.5  # medium confidence
            print("⚠️ No funding fields detected, returning Safe with caution.")
            return jsonify({
                "prediction": "Safe",
                "probability": probability,
                "status": "success",
                "explanation": explanation
            })



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

        print("🧪 Cleaned input data:", input_data)

        df = pd.DataFrame([input_data])

        # encode
        for col in label_encoders:
            if col in df.columns:
                try:
                    df[col] = label_encoders[col].transform(df[col].astype(str))
                except ValueError:
                    df[col] = 0
                    print(f"⚠️ Label encoding fallback: {col} → 0")

        df = df[feature_names]
        print("📊 Final DataFrame for model:\n", df)

        # Model predict
        probability = model.predict_proba(df)[0][1]
        prediction = model.predict(df)[0]

        explanation = []
        if business_age >= 10:
            explanation.append("The business has been operating for over 10 years, showing strong stability.")
        elif business_age >= 5:
            explanation.append("The business has a healthy age over 5 years, suggesting resilience.")

        if data.get("debt_financing", 0) > 0:
            explanation.append("Debt financing detected, which can show financial trust by lenders.")
        if data.get("seed", 0) > 0:
            explanation.append("Seed funding shows investor confidence at an early stage.")
        if data.get("venture", 0) > 0:
            explanation.append("Venture funding implies high growth potential.")
        if data.get("equity_crowdfunding", 0) > 0:
            explanation.append("Equity crowdfunding suggests public support.")

        if not explanation:
            explanation.append("Minimal history and funding detected, increasing risk.")

        print(f"✅ Prediction: {'Safe' if prediction==1 else 'Not Safe'} | Probability: {probability:.4f}")

        return jsonify({
            "prediction": "Safe" if prediction == 1 else "Not Safe",
            "probability": round(float(probability), 4),
            "status": "success",
            "explanation": explanation
        })

    except Exception as e:
        print("❌ Exception during prediction:", e)
        return jsonify({"error": str(e), "status": "failed"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "required_features": feature_names
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port, debug=False)
