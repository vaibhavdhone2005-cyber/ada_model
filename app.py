"""
Customer Churn Prediction — Flask Web App
Deploy target: Render.com

Loads a pre-trained scikit-learn AdaBoostClassifier (customer_churn_model.pkl)
and serves a form-based UI + JSON API to predict churn probability.
"""

import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "customer_churn_model.pkl")

model = None
model_load_error = None

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:  # noqa: BLE001
    model_load_error = str(e)

# Exact feature order the model was trained on (from model.feature_names_in_)
FEATURE_ORDER = [
    "Age",
    "Gender",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Subscription Type",
    "Contract Length",
    "Total Spend",
    "Last Interaction",
]

# ---------------------------------------------------------------------------
# Category encodings
# ---------------------------------------------------------------------------
# NOTE: The pickle file only contains the trained model, not the encoders
# used on the categorical columns during training. These mappings follow
# the standard alphabetical scikit-learn LabelEncoder convention, which is
# the common convention for this dataset. If your predictions look off,
# re-check these against your original training notebook and adjust below.
GENDER_MAP = {"Female": 0, "Male": 1}
SUBSCRIPTION_MAP = {"Basic": 0, "Premium": 1, "Standard": 2}
CONTRACT_MAP = {"Annual": 0, "Monthly": 1, "Quarterly": 2}


def build_feature_vector(payload: dict) -> np.ndarray:
    """Convert incoming form/JSON payload into an ordered numeric feature vector."""
    row = [
        float(payload["age"]),
        GENDER_MAP[payload["gender"]],
        float(payload["tenure"]),
        float(payload["usage_frequency"]),
        float(payload["support_calls"]),
        float(payload["payment_delay"]),
        SUBSCRIPTION_MAP[payload["subscription_type"]],
        CONTRACT_MAP[payload["contract_length"]],
        float(payload["total_spend"]),
        float(payload["last_interaction"]),
    ]
    return np.array(row, dtype=float).reshape(1, -1)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template(
        "index.html",
        genders=list(GENDER_MAP.keys()),
        subscriptions=list(SUBSCRIPTION_MAP.keys()),
        contracts=list(CONTRACT_MAP.keys()),
        model_ready=model is not None,
        model_load_error=model_load_error,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": f"Model not loaded: {model_load_error}"}), 500

    try:
        payload = request.get_json(force=True) if request.is_json else request.form
        features = build_feature_vector(payload)

        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        churn_probability = float(probabilities[1])  # class 1 = churn
        retain_probability = float(probabilities[0])

        if churn_probability >= 0.7:
            risk_level = "High"
        elif churn_probability >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return jsonify(
            {
                "prediction": prediction,
                "will_churn": bool(prediction == 1),
                "churn_probability": round(churn_probability * 100, 2),
                "retain_probability": round(retain_probability * 100, 2),
                "risk_level": risk_level,
            }
        )
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 400


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
