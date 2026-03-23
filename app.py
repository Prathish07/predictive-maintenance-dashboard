import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import shap
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import time

# ===== CONFIG =====
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

MODEL_DIR = r"G:\.VIT\SET_PROJECT\PredMaint\Models"
DATA_PATH = r"G:\.VIT\SET_PROJECT\PredMaint\Data\NASA_CMAPSS\train_FD001.txt"

# ===== LOAD MODELS =====
@st.cache_resource
def load_models():
    lstm = load_model(os.path.join(MODEL_DIR, "final_lstm.h5"))
    rf = joblib.load(os.path.join(MODEL_DIR, "final_rf_pron.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "cmapss_scaler.joblib"))
    xgb = joblib.load(os.path.join(MODEL_DIR, "final_xgb.joblib"))
    return lstm, rf, scaler, xgb

lstm_model, rf_model, scaler, xgb_model = load_models()

# ===== LOAD DATA =====
@st.cache_data
def load_data():
    col_names = [
        'unit_number', 'time_in_cycles',
        'op_setting_1', 'op_setting_2', 'op_setting_3'
    ] + [f'sensor_{i}' for i in range(1,22)]

    df = pd.read_csv(DATA_PATH, sep='\s+', header=None)
    df.columns = col_names
    return df

raw_df = load_data()
sensor_cols = [c for c in raw_df.columns if c.startswith("sensor_")]

# ===== LOAD FEATURE DATA (for SHAP + batch) =====
@st.cache_data
def load_feature_data():
    return pd.read_csv(os.path.join(MODEL_DIR, "cmapss_windows.csv"))

feature_df = load_feature_data()

# ===== TITLE =====
st.title("Predictive Maintenance Dashboard")
st.write("AI-driven Remaining Useful Life Prediction and Fault Monitoring System")

# ===== TABS =====
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "RUL Prediction",
    "Fault Detection",
    "Batch Prediction",
    "Explainability",
    "Real-Time Simulation"
])

# ==============================
# TAB 1 — RUL
# ==============================
with tab1:
    st.subheader("Remaining Useful Life Prediction")

    engine_id = st.number_input(
        "Select Engine ID",
        min_value=1,
        max_value=int(raw_df['unit_number'].max()),
        value=1
    )

    threshold = st.slider("Failure Threshold (RUL)", 5, 50, 30)

    engine_data = raw_df[raw_df['unit_number'] == engine_id].sort_values("time_in_cycles")
    current_cycle = engine_data['time_in_cycles'].max()

    if st.button("Predict RUL"):

        seq = engine_data[sensor_cols].tail(30).values
        seq = scaler.transform(seq)
        seq = seq.reshape(1, 30, len(sensor_cols))

        pred = lstm_model.predict(seq)[0][0]

        predicted_failure_cycle = current_cycle + pred

        st.success(f"Predicted RUL: {pred:.2f} cycles")
        st.info(f"Current Cycle: {current_cycle}")
        st.info(f"Predicted Failure Cycle: {int(predicted_failure_cycle)}")
        st.info(f"Remaining Cycles: {int(pred)}")

        # Risk classification
        if pred < threshold * 0.5:
            st.error("CRITICAL: Immediate maintenance required")
        elif pred < threshold:
            st.warning("WARNING: Maintenance recommended soon")
        else:
            st.success("SAFE: System operating normally")

    # ===== GRAPH =====
    if st.button("Show RUL Trend"):

        engine_data = engine_data.reset_index(drop=True)

        seq_len = 30
        actual, lstm_preds = [], []

        max_cycle = engine_data['time_in_cycles'].max()

        for i in range(seq_len, len(engine_data)):
            seq = engine_data[sensor_cols].iloc[i-seq_len:i].values
            seq = scaler.transform(seq)
            seq = seq.reshape(1, seq_len, len(sensor_cols))

            lstm_pred = lstm_model.predict(seq, verbose=0)[0][0]
            lstm_preds.append(lstm_pred)

            actual.append(max_cycle - engine_data.loc[i, 'time_in_cycles'])

        rmse_lstm = np.sqrt(mean_squared_error(actual, lstm_preds))

        fig, ax = plt.subplots()
        ax.plot(actual, label="Actual")
        ax.plot(lstm_preds, label=f"LSTM (RMSE={rmse_lstm:.2f})")
        ax.axhline(y=threshold, linestyle='--', label="Failure Threshold")

        ax.legend()
        ax.set_title("RUL Prediction (LSTM)")

        st.pyplot(fig)

# ==============================
# TAB 2 — FAULT
# ==============================
with tab2:
    st.subheader("Fault Detection")

    # ===== FAULT INFO =====
    fault_info = {
        0: {"name": "Normal Operation", "severity": "Low",
            "action": "No maintenance required. Continue monitoring."},
        1: {"name": "Inner Race Fault", "severity": "Medium",
            "action": "Inspect inner race. Schedule maintenance."},
        2: {"name": "Outer Race Fault", "severity": "Medium",
            "action": "Check outer race wear. Plan maintenance."},
        3: {"name": "Ball Fault", "severity": "Medium",
            "action": "Inspect rolling elements. Lubricate or replace."},
        4: {"name": "Combined Fault", "severity": "High",
            "action": "Multiple faults detected. Immediate inspection required."},
        5: {"name": "Severe Degradation", "severity": "Critical",
            "action": "Immediate shutdown and replacement required."}
    }

    st.markdown("### Fault Classes and Maintenance Actions")
    for k, v in fault_info.items():
        st.write(f"{k} → {v['name']} ({v['severity']})")

    # ===== INPUT =====
    inputs = []
    for i in range(min(6, rf_model.n_features_in_)):
        val = st.number_input(f"Feature {i}", value=0.5, key=f"fault_{i}")
        inputs.append(val)

    if len(inputs) < rf_model.n_features_in_:
        inputs += [0.0] * (rf_model.n_features_in_ - len(inputs))

    # ===== PREDICTION =====
    if st.button("Predict Fault"):
        pred = rf_model.predict(np.array(inputs).reshape(1, -1))[0]

        info = fault_info.get(pred)

        st.subheader(f"Predicted Fault: {info['name']}")

        if info["severity"] == "Low":
            st.success(f"Severity: {info['severity']}")
        elif info["severity"] == "Medium":
            st.warning(f"Severity: {info['severity']}")
        else:
            st.error(f"Severity: {info['severity']}")

        st.markdown("### Recommended Action")
        st.write(info["action"])

# ==============================
# TAB 3 — BATCH
# ==============================
with tab3:
    st.subheader("Batch Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.write(df.head())

        if st.button("Run Prediction"):
            drop_cols = ['unit_number', 'end_cycle', 'RUL', 'RUL_cap']
            df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns])

            feature_cols = feature_df.drop(columns=[c for c in drop_cols if c in feature_df.columns]).columns
            df_clean = df_clean.reindex(columns=feature_cols, fill_value=0)

            preds = xgb_model.predict(df_clean.values)
            df["Predicted_RUL"] = preds

            st.dataframe(df.head())

# ==============================
# TAB 4 — SHAP
# ==============================
with tab4:
    st.subheader("Explainability (SHAP)")

    try:
        feature_cols = feature_df.drop(columns=['unit_number','end_cycle','RUL','RUL_cap'], errors='ignore').columns

        sample = pd.DataFrame(
            np.random.rand(1, len(feature_cols)),
            columns=feature_cols
        )

        explainer = shap.Explainer(xgb_model)
        shap_values = explainer(sample)

        plt.figure()
        shap.plots.bar(shap_values, show=False)
        st.pyplot(plt.gcf())

    except Exception as e:
        st.error(str(e))

# ==============================
# TAB 5 — REAL-TIME
# ==============================
with tab5:
    st.subheader("Real-Time Sensor Simulation")

    if st.button("Start Simulation"):
        chart = st.line_chart()

        for i in range(50):
            val = 120 - i + np.random.randn() * 5
            chart.add_rows(pd.DataFrame([val]))
            time.sleep(0.1)

# ===== FOOTER =====
st.markdown("---")
st.write("Final System: AI-Based Predictive Maintenance with Explainability and Maintenance Decision Support")