import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

# ---------------------------------------------------------
# Load the trained pipeline (cached so it only loads once)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/pipeline.pkl")

pipeline = load_model()

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("🚢 Titanic Survival Predictor")
st.write(
    "Enter a passenger's details below to predict whether they would have "
    "survived the Titanic disaster."
)

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        options=[1, 2, 3],
        format_func=lambda x: {1: "1st Class", 2: "2nd Class", 3: "3rd Class"}[x],
    )
    sex = st.selectbox("Sex", options=["male", "female"])

with col2:
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
    fare = st.number_input("Fare ($)", min_value=0.0, max_value=600.0, value=32.0, step=1.0)

st.divider()

if st.button("Predict Survival", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        "pclass": pclass,
        "sex": sex,
        "age": age,
        "fare": fare,
    }])

    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0]

    survived = int(prediction) == 1
    survival_prob = probability[1]
    no_survival_prob = probability[0]

    if survived:
        st.success(f"### ✅ Likely to SURVIVE")
    else:
        st.error(f"### ❌ Likely to NOT survive")

    c1, c2 = st.columns(2)
    c1.metric("Survival Probability", f"{survival_prob * 100:.1f}%")
    c2.metric("No Survival Probability", f"{no_survival_prob * 100:.1f}%")

    st.progress(float(survival_prob))

st.divider()
st.caption(
    "Model: scikit-learn pipeline trained on the Titanic dataset "
    "(Pclass, Sex, Age, Fare). For educational purposes only."
)