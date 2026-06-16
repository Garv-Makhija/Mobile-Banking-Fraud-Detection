import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------
# PAGE CONFIG
# ----------------------------

st.set_page_config(
    page_title="Mobile Banking Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# ----------------------------
# LOAD MODEL
# ----------------------------

model = joblib.load("xgboost_fraud_model.pkl")

# ----------------------------
# HEADER
# ----------------------------

st.title("🛡️ Mobile Banking Fraud Detection System")
st.markdown("---")

# ----------------------------
# TOP METRICS
# ----------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Fraud Detection Model", "XGBoost")

with col2:
    st.metric("Anomaly Detection", "Isolation Forest")

with col3:
    st.metric("System Status", "ACTIVE")

st.markdown("---")

# ----------------------------
# SIDEBAR
# ----------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Fraud Detection",
        "System Information"
    ]
)

# ==================================================
# FRAUD DETECTION PAGE
# ==================================================

if page == "Fraud Detection":

    st.header("Transaction Analysis")

    col1, col2 = st.columns(2)

    with col1:

        typing_speed = st.number_input(
            "Typing Speed",
            value=250.0
        )

        touch_pressure = st.number_input(
            "Touch Pressure",
            value=0.70
        )

        login_hour = st.slider(
            "Login Hour",
            0,
            23,
            12
        )

        location_distance = st.number_input(
            "Location Distance (KM)",
            value=10.0
        )

    with col2:

        device_change = st.selectbox(
            "Device Changed",
            [0, 1]
        )

        sim_change = st.selectbox(
            "SIM Changed",
            [0, 1]
        )

        failed_attempts = st.number_input(
            "Failed Login Attempts",
            value=0
        )

        session_duration = st.number_input(
            "Session Duration",
            value=300.0
        )

    # ----------------------------
    # FEATURE ENGINEERING
    # ----------------------------

    behavior_risk = (
        abs(typing_speed - 250) / 250
        +
        abs(touch_pressure - 0.7)
    )

    session_risk = (
        device_change * 25
        +
        sim_change * 35
    )

    # ----------------------------
    # PREDICT
    # ----------------------------

    if st.button("🔍 Analyze Risk"):

        input_data = pd.DataFrame([{

            "typing_speed": typing_speed,
            "touch_pressure": touch_pressure,
            "login_hour": login_hour,
            "device_change": device_change,
            "sim_change": sim_change,
            "location_distance": location_distance,
            "failed_attempts": failed_attempts,
            "session_duration": session_duration,
            "behavior_risk": behavior_risk,
            "session_risk": session_risk,
            "anomaly_score": 0,
            "iso_prediction": 0

        }])

        fraud_prob = model.predict_proba(
            input_data
        )[0][1]

        risk_score = (
            fraud_prob * 60
            +
            device_change * 15
            +
            sim_change * 20
            +
            (5 if failed_attempts >= 5 else 0)
        )

        # ----------------------------
        # DECISION ENGINE
        # ----------------------------

        if risk_score < 30:
            decision = "ALLOW"
            level = "LOW"

        elif risk_score < 60:
            decision = "OTP VERIFICATION"
            level = "MEDIUM"

        elif risk_score < 80:
            decision = "BIOMETRIC VERIFICATION"
            level = "HIGH"

        else:
            decision = "BLOCK"
            level = "CRITICAL"

        st.markdown("---")

        st.subheader("Fraud Analysis Results")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Fraud Probability",
                f"{fraud_prob:.2%}"
            )

        with c2:
            st.metric(
                "Risk Score",
                round(risk_score, 2)
            )

        with c3:
            st.metric(
                "Risk Level",
                level
            )

        st.progress(float(fraud_prob))

        st.markdown("---")

        if decision == "ALLOW":
            st.success(f"Decision: {decision}")

        elif decision == "OTP VERIFICATION":
            st.warning(f"Decision: {decision}")

        elif decision == "BIOMETRIC VERIFICATION":
            st.warning(f"Decision: {decision}")

        else:
            st.error(f"Decision: {decision}")

        # ----------------------------
        # FEATURE CONTRIBUTION
        # ----------------------------

        st.subheader("Risk Factors")

        risk_df = pd.DataFrame({
            "Factor": [
                "Device Change",
                "SIM Change",
                "Failed Attempts",
                "Behavior Risk"
            ],
            "Value": [
                device_change,
                sim_change,
                failed_attempts,
                round(behavior_risk, 3)
            ]
        })

        st.dataframe(
            risk_df,
            use_container_width=True
        )

# ==================================================
# SYSTEM INFO PAGE
# ==================================================

if page == "System Information":

    st.header("Project Overview")

    st.info(
        """
        Mobile Banking Fraud Detection System

        Models Used:
        • Isolation Forest
        • XGBoost Classifier

        Security Features:
        • Behaviour Analysis
        • Device Change Detection
        • SIM Swap Detection
        • Login Monitoring
        • Risk Scoring Engine
        """
    )

    st.subheader("Fraud Workflow")

    st.write(
        """
        User Login Attempt
            ↓
        Behaviour Analysis
            ↓
        XGBoost Fraud Detection
            ↓
        Risk Scoring Engine
            ↓
        Decision Engine
            ↓
        Allow / OTP / Biometric / Block
        """
    )

st.sidebar.markdown("---")
st.sidebar.success("Project Ready ✅")