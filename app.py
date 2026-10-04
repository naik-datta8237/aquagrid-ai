import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="AquaGrid AI — SCADA Predictive Maintenance Engine",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# BRAND CUSTOM CSS
# ---------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #F8FAFC;
        color: #122B44;
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4 {
        color: #122B44 !important;
        font-weight: 700;
    }

    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid rgba(18, 43, 68, 0.08);
    }

    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #122B44;
    }

    [data-testid="stMetricLabel"] {
        color: #218B98;
        font-weight: 600;
        font-size: 13px;
        text-transform: uppercase;
    }

    .stButton>button {
        background-color: #05C2D1;
        color: #FFFFFF;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: all 0.2s ease;
    }

    .stButton>button:hover {
        background-color: #218B98;
        color: #FFFFFF;
        border: none;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# DATASET + MODEL
# ---------------------------------------------------------
DATASET_PATH = Path(__file__).parent / "telemetry_dataset.csv"

FEATURES = [
    "Pressure (bar)",
    "Flow Rate (L/s)",
    "Temperature (°C)"
]

REQUIRED_COLUMNS = [
    "Timestamp",
    "Sensor_ID",
    "Pressure (bar)",
    "Flow Rate (L/s)",
    "Temperature (°C)",
    "Leak Status",
    "Burst Status"
]


@st.cache_resource
def load_and_train_model():

    # Load actual SCADA dataset
    df = pd.read_csv(DATASET_PATH)

    # Validate dataset structure
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    # Convert model features to numeric
    for column in FEATURES:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Remove rows with missing model inputs
    df = df.dropna(subset=FEATURES).copy()

    # -----------------------------------------------------
    # FEATURE MATRIX
    # Only physical telemetry variables are used.
    # Sensor_ID is an identifier, not an ML feature.
    # Leak/Burst status are evaluation labels.
    # -----------------------------------------------------
    X = df[FEATURES]

    # Standardize telemetry
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------------------------------------
    # ACTUAL ISOLATION FOREST MODEL
    # -----------------------------------------------------
    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42
    )

    model.fit(X_scaled)

    # Training decision scores
    training_scores = model.decision_function(X_scaled)

    # Sort scores for relative risk calculation
    sorted_scores = np.sort(training_scores)

    # Model predictions
    df["AI Prediction"] = model.predict(X_scaled)

    # -1 = anomaly
    #  1 = normal
    df["AI Anomaly"] = df["AI Prediction"] == -1

    return df, scaler, model, sorted_scores


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
try:
    df, scaler, model, sorted_scores = load_and_train_model()

except Exception as e:
    st.error(f"Unable to load the SCADA dataset or train the AI model: {e}")
    st.stop()


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def calculate_risk_score(decision_score):
    """
    Converts the Isolation Forest decision score into a
    relative 0–100 anomaly risk score.

    Lower Isolation Forest scores indicate more anomalous
    observations.

    This is NOT a probability.
    """

    percentile = (
        np.searchsorted(
            sorted_scores,
            decision_score,
            side="right"
        ) / len(sorted_scores)
    )

    risk = (1 - percentile) * 100

    return float(np.clip(risk, 0, 100))


def classify_risk(risk_score):

    if risk_score >= 90:
        return (
            "🔴 CRITICAL ANOMALY DETECTED",
            "critical"
        )

    elif risk_score >= 70:
        return (
            "🟠 WARNING — Operational Variance Detected",
            "warning"
        )

    else:
        return (
            "🟢 SYSTEM NOMINAL",
            "normal"
        )


def safe_numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce"
    )


# ---------------------------------------------------------
# HEADER SECTION
# ---------------------------------------------------------
st.title(
    "💧 AquaGrid AI — SCADA Predictive Maintenance Engine"
)

st.caption(
    "Smarter Networks. Safer Water. — "
    "AI-Assisted Municipal SCADA Intelligence"
)

st.markdown("---")


# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.markdown("### 👤 SCADA Control Panel")

# Use actual Sensor IDs from the dataset
sensor_options = ["All Sensors"] + sorted(
    df["Sensor_ID"].astype(str).unique().tolist()
)

sensor_node = st.sidebar.selectbox(
    "Select Sensor Node",
    sensor_options
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    "### ⚙️ Live Parameter Simulation"
)

st.sidebar.info(
    "Adjust the three SCADA telemetry parameters "
    "used by the current Isolation Forest MVP."
)


# ---------------------------------------------------------
# DATASET BASELINES
# ---------------------------------------------------------
default_pressure = float(
    df["Pressure (bar)"].median()
)

default_flow = float(
    df["Flow Rate (L/s)"].median()
)

default_temperature = float(
    df["Temperature (°C)"].median()
)


# ---------------------------------------------------------
# LIVE PARAMETER SLIDERS
# ---------------------------------------------------------
pressure = st.sidebar.slider(
    "Hydraulic Pressure (bar)",
    min_value=0.0,
    max_value=6.0,
    value=round(default_pressure, 2),
    step=0.05
)

flow_rate = st.sidebar.slider(
    "Flow Rate (L/s)",
    min_value=0.0,
    max_value=250.0,
    value=round(default_flow, 1),
    step=1.0
)

temperature = st.sidebar.slider(
    "Temperature (°C)",
    min_value=0.0,
    max_value=40.0,
    value=round(default_temperature, 1),
    step=0.5
)


# ---------------------------------------------------------
# CURRENT TELEMETRY
# ---------------------------------------------------------
current_data = pd.DataFrame({
    "Pressure (bar)": [pressure],
    "Flow Rate (L/s)": [flow_rate],
    "Temperature (°C)": [temperature]
})


# ---------------------------------------------------------
# ISOLATION FOREST INFERENCE
# ---------------------------------------------------------
current_scaled = scaler.transform(
    current_data[FEATURES]
)

prediction = model.predict(current_scaled)[0]

decision_score = model.decision_function(
    current_scaled
)[0]

anomaly_risk = calculate_risk_score(
    decision_score
)

op_status, status_type = classify_risk(
    anomaly_risk
)


# ---------------------------------------------------------
# OPERATIONAL STATUS
# ---------------------------------------------------------
if status_type == "critical":

    st.error(
        f"**Operational Status:** {op_status}"
    )

elif status_type == "warning":

    st.warning(
        f"**Operational Status:** {op_status}"
    )

else:

    st.success(
        f"**Operational Status:** {op_status}"
    )


# ---------------------------------------------------------
# MAIN METRIC CARDS
# ---------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)


# Dataset baselines
baseline_pressure = df["Pressure (bar)"].median()
baseline_flow = df["Flow Rate (L/s)"].median()
baseline_temperature = df["Temperature (°C)"].median()


with col1:

    pressure_diff = (
        pressure - baseline_pressure
    )

    st.metric(
        label="Hydraulic Pressure",
        value=f"{pressure:.2f} bar",
        delta=(
            f"{pressure_diff:+.2f} bar vs dataset baseline"
        )
    )


with col2:

    flow_diff = (
        flow_rate - baseline_flow
    )

    st.metric(
        label="Flow Rate",
        value=f"{flow_rate:.1f} L/s",
        delta=f"{flow_diff:+.1f} L/s"
    )


with col3:

    temperature_diff = (
        temperature - baseline_temperature
    )

    st.metric(
        label="Temperature",
        value=f"{temperature:.1f} °C",
        delta=f"{temperature_diff:+.1f} °C"
    )


with col4:

    st.metric(
        label="AI Prediction",
        value=(
            "ANOMALY"
            if prediction == -1
            else "NORMAL"
        ),
        delta=(
            "Isolation Forest"
        )
    )


with col5:

    st.metric(
        label="Anomaly Risk Score",
        value=f"{anomaly_risk:.1f}/100",
        delta=(
            "HIGH RISK"
            if anomaly_risk >= 70
            else "NOMINAL"
        ),
        delta_color=(
            "inverse"
            if anomaly_risk >= 70
            else "normal"
        )
    )


st.markdown("---")


# ---------------------------------------------------------
# CURRENT TELEMETRY DETAILS
# ---------------------------------------------------------
st.subheader("📡 Current SCADA Telemetry")

telemetry_col1, telemetry_col2 = st.columns(2)

with telemetry_col1:

    st.dataframe(
        current_data,
        use_container_width=True,
        hide_index=True
    )

with telemetry_col2:

    st.markdown("### 🧠 AI Decision")

    st.write(
        f"**Isolation Forest Decision Score:** "
        f"`{decision_score:.4f}`"
    )

    st.write(
        f"**Relative Anomaly Risk:** "
        f"`{anomaly_risk:.1f}/100`"
    )

    if prediction == -1:

        st.warning(
            "The current telemetry pattern is "
            "classified as an anomaly by the Isolation Forest model."
        )

    else:

        st.success(
            "The current telemetry pattern is "
            "classified as normal by the Isolation Forest model."
        )


st.markdown("---")


# ---------------------------------------------------------
# DATASET INFORMATION
# ---------------------------------------------------------
with st.expander("📊 SCADA Dataset & Model Information"):

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.metric(
            "Telemetry Records",
            f"{len(df):,}"
        )

    with info_col2:
        st.metric(
            "Sensor Nodes",
            f"{df['Sensor_ID'].nunique():,}"
        )

    with info_col3:
        st.metric(
            "AI Features",
            "3"
        )

    st.markdown(
        """
        **Current AI model inputs:**
        - Pressure (bar)
        - Flow Rate (L/s)
        - Temperature (°C)

        **Evaluation/context fields:**
        - Leak Status
        - Burst Status

        `Sensor_ID` is treated as an identifier and is not used as
        an ML feature.

        The current MVP uses an **Isolation Forest** trained on the
        available synthetic SCADA telemetry dataset.
        """
    )


# ---------------------------------------------------------
# INTERACTIVE SCADA AI TELEMETRY ASSISTANT
# ---------------------------------------------------------
st.subheader(
    "🤖 SCADA AI Telemetry Assistant"
)

query_option = st.selectbox(
    "Select a SCADA Query to Run:",
    [
        "1. What is the baseline average hydraulic pressure across operational zones?",
        "2. Is there an active telemetry anomaly detected in the network?",
        "3. What is the estimated water loss rate and financial risk (NRW)?",
        "4. Which sensor requires immediate preventive maintenance?",
        "5. Generate an automated dispatch ticket for field engineers."
    ]
)


# ---------------------------------------------------------
# EXECUTE QUERY
# ---------------------------------------------------------
if st.button("Execute AI Analysis"):

    st.markdown(
        "### 📋 AI Diagnostic Output"
    )

    # -----------------------------------------------------
    # QUERY 1
    # -----------------------------------------------------
    if "1." in query_option:

        st.write(
            f"""
            **System Baseline Report:**

            Based on the current SCADA dataset, the median
            hydraulic pressure is **{baseline_pressure:.2f} bar**.

            Current measured pressure is
            **{pressure:.2f} bar**.

            **Variance:** `{pressure_diff:+.2f} bar`
            """
        )

        st.info(
            "The baseline is calculated directly from the "
            "available SCADA telemetry dataset."
        )


    # -----------------------------------------------------
    # QUERY 2
    # -----------------------------------------------------
    elif "2." in query_option:

        st.write(
            f"""
            **Isolation Forest Anomaly Analysis**

            - **AI Prediction:** `{
                'ANOMALY'
                if prediction == -1
                else 'NORMAL'
            }`
            - **Anomaly Risk Score:** `{anomaly_risk:.1f}/100`
            - **Decision Score:** `{decision_score:.4f}`
            - **Operational Status:** `{op_status}`
            """
        )

        if prediction == -1:

            st.error(
                "The current combination of pressure, flow and "
                "temperature is outside the model's learned "
                "normal operating patterns."
            )

        else:

            st.success(
                "No Isolation Forest anomaly was detected "
                "for the current telemetry combination."
            )


    # -----------------------------------------------------
    # QUERY 3
    # -----------------------------------------------------
    elif "3." in query_option:

        # Prototype estimate only
        estimated_loss_lps = round(
            anomaly_risk / 100 * 18.5,
            2
        )

        financial_loss = int(
            estimated_loss_lps * 3000
        )

        st.write(
            f"""
            **Non-Revenue Water (NRW) Impact Analysis**

            - **Relative Anomaly Risk:** `{anomaly_risk:.1f}/100`
            - **Prototype Estimated Water Loss:** `{estimated_loss_lps} L/s`
            - **Prototype Revenue Impact:** `₹{financial_loss:,} / day`
            """
        )

        st.warning(
            "NRW and financial values are prototype estimates "
            "for the MVP demonstration. They are not directly "
            "predicted by the current Isolation Forest model."
        )


    # -----------------------------------------------------
    # QUERY 4
    # -----------------------------------------------------
    elif "4." in query_option:

        selected_sensor = (
            sensor_node
            if sensor_node != "All Sensors"
            else "Current SCADA Pattern"
        )

        st.write(
            f"""
            **Predictive Maintenance Assessment**

            - **Target Sensor:** `{selected_sensor}`
            - **AI Anomaly Risk:** `{anomaly_risk:.1f}/100`
            - **AI Classification:** {
                'ANOMALY'
                if prediction == -1
                else 'NORMAL'
            }
            """
        )

        if anomaly_risk >= 90:

            st.error(
                "Immediate inspection is recommended. "
                "The current telemetry pattern is highly anomalous."
            )

        elif anomaly_risk >= 70:

            st.warning(
                "Preventive inspection is recommended "
                "because the telemetry pattern shows elevated risk."
            )

        else:

            st.success(
                "No immediate preventive maintenance action "
                "is indicated by the current AI assessment."
            )


    # -----------------------------------------------------
    # QUERY 5
    # -----------------------------------------------------
    elif "5." in query_option:

        if prediction == -1 or anomaly_risk >= 70:

            sensor_label = (
                sensor_node
                if sensor_node != "All Sensors"
                else "Selected SCADA Sensor"
            )

            work_order_id = (
                f"WO-{np.random.randint(1000, 9999)}"
            )

            st.success(
                f"""
                **AUTOMATED WORK ORDER GENERATED (#{work_order_id})**

                - **Sensor:** {sensor_label}
                - **Trigger:** Anomaly Risk {anomaly_risk:.1f}/100
                - **AI Classification:** ANOMALY
                - **Priority:** {
                    'CRITICAL'
                    if anomaly_risk >= 90
                    else 'HIGH'
                }
                - **Recommended Action:** Inspect the affected
                  network segment and verify pressure/flow conditions.
                """
            )

        else:

            st.info(
                "No work order generated because the current "
                "telemetry pattern is within the model's normal range."
            )


# ---------------------------------------------------------
# MODEL LIMITATION / MVP DISCLAIMER
# ---------------------------------------------------------
st.markdown("---")

st.info(
    """
    **MVP Model Note:** AquaGrid AI currently demonstrates an
    Isolation Forest anomaly-detection workflow using synthetic
    SCADA telemetry. The model uses pressure, flow rate and
    temperature as inputs. Production deployment would require
    historical municipal SCADA data, model calibration and
    operational validation before using predictions for real
    maintenance decisions.
    """
)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")

st.caption(
    "AquaGrid AI Platform • SCADA Anomaly Detection MVP • "
    "Streamlit + Python + scikit-learn"
)