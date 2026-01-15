import time
import pandas as pd
import streamlit as st
import joblib
from log_utils import log_prediction

st.set_page_config(page_title="Insurance Charges Predictor", layout="centered")
st.title("Insurance Charges Prediction Dashboard")

# Load models
@st.cache_resource
def load_models():
    baseline_model = joblib.load("model_v1.pkl")
    improved_model = joblib.load("model_v2.pkl")
    return baseline_model, improved_model

baseline_model, improved_model = load_models()

# Session state for predictions
if "pred_ready" not in st.session_state:
    st.session_state["pred_ready"] = False

# Input parameters
st.sidebar.header("Input Parameters")
age = st.sidebar.slider("Age", int(18), int(65), 30)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
bmi = st.sidebar.slider("BMI", 15.0, 53.0, 25.0)
children = st.sidebar.slider("Children", 0, 5, 0)
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

input_df = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region]
})

st.subheader("Input Summary")
st.write(input_df)

# Run prediction
if st.button("Run Prediction"):
    start_time = time.time()
    baseline_pred = baseline_model.predict(input_df)[0]
    improved_pred = improved_model.predict(input_df)[0]
    latency_ms = (time.time() - start_time) * 1000.0

    st.session_state["baseline_pred"] = baseline_pred
    st.session_state["improved_pred"] = improved_pred
    st.session_state["latency_ms"] = latency_ms
    st.session_state["pred_ready"] = True

# Display predictions
if st.session_state.get("pred_ready"):
    st.subheader("Predictions")
    st.write(f"Baseline Model Prediction: ${st.session_state['baseline_pred']:.2f}")
    st.write(f"Improved Model Prediction: ${st.session_state['improved_pred']:.2f}")
    st.write(f"Latency: {st.session_state['latency_ms']:.1f} ms")

# ---------- FEEDBACK SECTION ----------
st.subheader("Your Feedback on These Predictions")

feedback_score = st.slider(
    "How useful were these predictions? (1 = Poor, 5 = Excellent)",
    min_value=1,
    max_value=5,
    value=4,
    key="feedback_score",
)
feedback_text = st.text_area("Comments (optional)", key="feedback_text")

# ---------- BUTTON 2: SUBMIT FEEDBACK ----------
if st.button("Submit Feedback"):
    if not st.session_state["pred_ready"]:
        st.warning("Please run the prediction first, then submit your feedback.")
    else:
        # Log both models using saved predictions and input summary
        log_prediction(
            model_version="v1_old",
            model_type="baseline",
            input_summary=input_df.to_dict(orient="records")[0],
            prediction=st.session_state['baseline_pred'],
            latency_ms=st.session_state['latency_ms'],
            feedback_score=feedback_score,
            feedback_text=feedback_text
        )

        # Log the new model (v2)
        
        log_prediction(
            model_version="v2_new",
            model_type="improved",
            input_summary=input_df.to_dict(orient="records")[0],
            prediction=st.session_state['improved_pred'],
            latency_ms=st.session_state['latency_ms'],
            feedback_score=feedback_score,
            feedback_text=feedback_text
        )
        st.success(
            "Feedback and predictions have been saved to monitoring_logs.csv. "
            "You can now view them in the monitoring dashboard."
        )