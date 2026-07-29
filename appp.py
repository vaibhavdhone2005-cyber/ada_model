import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Analytics & Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Executive Styling (CSS)
# ---------------------------------------------------------
st.markdown("""
    <style>
    /* Executive Color Scheme & Clean Layout */
    .main {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    .stApp {
        background-color: #0F172A;
    }
    
    /* Header Styling */
    .header-box {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 25px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }
    .header-title {
        color: #38BDF8;
        font-size: 32px;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        color: #94A3B8;
        font-size: 15px;
        margin-top: 5px;
    }

    /* Metric Card Styling */
    .metric-card {
        background: #1E293B;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #38BDF8;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    
    /* Custom Button */
    .stButton>button {
        background: linear-gradient(90deg, #0284C7 0%, #0369A1 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 12px 28px;
        border-radius: 8px;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #0369A1 0%, #075985 100%);
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    with open("ada_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading `ada_model.pkl`: {e}")
    st.stop()

# ---------------------------------------------------------
# Header & Presentation Intro
# ---------------------------------------------------------
st.markdown("""
    <div class="header-box">
        <h1 class="header-title">Enterprise Churn Intelligence Platform</h1>
        <p class="header-subtitle">Real-time Predictive Analytics Powered by AdaBoost Classifier</p>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar - Inputs Form
# ---------------------------------------------------------
st.sidebar.header("📋 Customer Profile Inputs")

with st.sidebar.form("customer_input_form"):
    st.subheader("Demographics & Usage")
    age = st.slider("Age", min_value=18, max_value=90, value=35)
    gender = st.selectbox("Gender", options=["Female", "Male"])
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=12)
    usage_freq = st.slider("Usage Frequency (Days/Month)", min_value=1, max_value=30, value=15)
    
    st.subheader("Service & Engagement")
    support_calls = st.number_input("Support Calls", min_value=0, max_value=20, value=2)
    payment_delay = st.number_input("Payment Delay (Days)", min_value=0, max_value=60, value=3)
    subscription_type = st.selectbox("Subscription Type", options=["Basic", "Standard", "Premium"])
    contract_length = st.selectbox("Contract Length", options=["Monthly", "Quarterly", "Annual"])
    total_spend = st.number_input("Total Spend ($)", min_value=0.0, max_value=10000.0, value=500.0, step=50.0)
    last_interaction = st.number_input("Last Interaction (Days Ago)", min_value=0, max_value=180, value=10)
    
    submit_btn = st.form_submit_button("🔍 Run Churn Assessment")

# Feature Encodings (Map choices to categorical numerical format)
gender_encoded = 1 if gender == "Male" else 0

sub_map = {"Basic": 0, "Standard": 1, "Premium": 2}
subscription_encoded = sub_map[subscription_type]

contract_map = {"Monthly": 0, "Quarterly": 1, "Annual": 2}
contract_encoded = contract_map[contract_length]

# Feature Array matching exact order:
# ['Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Subscription Type', 'Contract Length', 'Total Spend', 'Last Interaction']
features = np.array([[
    age, gender_encoded, tenure, usage_freq, support_calls,
    payment_delay, subscription_encoded, contract_encoded, total_spend, last_interaction
]])

# ---------------------------------------------------------
# Dashboard Body / Results Display
# ---------------------------------------------------------
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📌 Customer Profile Summary")
    df_summary = pd.DataFrame({
        "Attribute": ["Age", "Gender", "Tenure", "Usage Frequency", "Support Calls", "Payment Delay", "Subscription", "Contract", "Total Spend", "Last Interaction"],
        "Value": [f"{age} yrs", gender, f"{tenure} mos", f"{usage_freq} days/mo", support_calls, f"{payment_delay} days", subscription_type, contract_length, f"${total_spend:,.2f}", f"{last_interaction} days ago"]
    })
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

with col2:
    st.subheader("🎯 Model Risk Assessment")
    
    # Run Prediction
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    churn_probability = round(float(probabilities[1]) * 100, 2)
    
    # Gauge Chart for Risk Score
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_probability,
        number={'suffix': "%"},
        title={'text': "Churn Probability Risk", 'font': {'size': 20, 'color': "#FFFFFF"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
            'bar': {'color': "#EF4444" if churn_probability >= 50 else "#10B981"},
            'bgcolor': "#1E293B",
            'borderwidth': 1,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(16, 185, 129, 0.2)'},
                {'range': [40, 70], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
            ],
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#FFFFFF"},
        height=260,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Executive Verdict Display
    if prediction == 1:
        st.error(f"⚠️ **High Churn Risk Detected ({churn_probability}%)**\n\nRecommendation: Trigger retention workflow immediately.")
    else:
        st.success(f"✅ **Low Churn Risk ({100 - churn_probability:.2f}% Retained)**\n\nRecommendation: Customer engagement is stable.")
