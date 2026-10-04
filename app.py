"""
Heart Disease Prediction - Student Demo App
----------------------------------------------------------
A small Streamlit web app that lets anyone interact with the trained
Support Vector Classifier from the "Heart_Disease_Prediction" notebook,
without needing to install Python or run Jupyter.

It loads the same svc_trained_model.pkl produced by that notebook.
All 4 input attributes (Sex, CP, Exang, Oldpeak) are already numeric, so
NO label encoding is needed - the values chosen in the web form go straight
to the model.

DISCLAIMER: educational project only - NOT a medical diagnosis tool.
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# ---------------------------------------------------------------------------
# Load the trained model (same file the notebook produced)
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).parent / "svc_trained_model.pkl"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


model = load_model()

# ---------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------

st.title("❤️ Heart Disease Prediction System")
st.markdown(
    """
    This is a teaching demo built from the **Heart Disease Prediction** notebook.
    Enter the patient's details below and the trained Support Vector Classifier
    (SVC) will predict whether heart disease is likely.

    *Model accuracy on held-out test data: **72%** (Cleveland dataset, 303 patients).*
    """
)

st.divider()

# Display text for each numeric code. The model receives the CODE (a number),
# the user just sees the friendly text - so no LabelEncoder is needed.
SEX_OPTIONS = {1: "Male", 0: "Female"}
CP_OPTIONS = {
    1: "Typical angina",
    2: "Atypical angina",
    3: "Non-anginal pain",
    4: "Asymptomatic",
}
EXANG_OPTIONS = {0: "No", 1: "Yes"}

col1, col2 = st.columns(2)
with col1:
    sex = st.selectbox("Sex", list(SEX_OPTIONS.keys()), format_func=SEX_OPTIONS.get)
    cp = st.selectbox("Chest Pain Type (CP)", list(CP_OPTIONS.keys()),
                      format_func=CP_OPTIONS.get, index=3)
with col2:
    exang = st.selectbox("Exercise Induced Angina (Exang)", list(EXANG_OPTIONS.keys()),
                         format_func=EXANG_OPTIONS.get)
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=7.0,
                              value=1.0, step=0.1)

if st.button("Predict Heart Disease", type="primary", use_container_width=True):
    # Feature vector: same column names and SAME ORDER as the training data
    user_input = pd.DataFrame({
        "Sex": [sex],
        "CP": [cp],
        "Exang": [exang],
        "Oldpeak": [oldpeak],
    })

    prediction = model.predict(user_input)[0]

    st.divider()
    if prediction == 1:
        st.error("### ⚠️ HEART DISEASE LIKELY")
    else:
        st.success("### ✅ HEART DISEASE NOT LIKELY")

    with st.expander("See the feature vector sent to the model"):
        st.dataframe(user_input, hide_index=True)

st.divider()
st.caption(
    "Built for teaching purposes from the Cleveland Heart Disease dataset using only "
    "4 input attributes. This is NOT a medical diagnosis tool - please consult a "
    "doctor for any health concern."
)
