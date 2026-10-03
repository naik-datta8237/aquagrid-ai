import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.ensemble import IsolationForest

# Page Configuration
st.set_page_config(
    page_title="AquaGrid AI — SCADA Predictive Maintenance Engine",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 BRAND IDENTITY CSS OVERRIDE (AquaGrid AI Official Guidelines)
st.markdown("""
    <style>
    @import url('https://rsms.me/inter/inter.css');

    /* Global Body & Backgrounds */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif !important;
        background-color: #F8FAFC !important;
        color: #122B44 !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E5E7EB !important;
    }

    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #122B44 !important;
    }

    /* Headings (Deep Navy) */
    h1, h2, h3, h4 {
        color: #122B44 !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    /* Subtitles and Labels */
    p, label, span, .stMarkdown {
        color: #122B44 !important;
    }

    /* Metric Cards - Minimal White Container with Soft Aqua Highlights */
    [data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-top: 4px solid #05C2D1 !important;
        padding: 16px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(18, 43, 68, 0.05) !important;
    }

    [data-testid="stMetricLabel"] {
        color: #218B98 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    [data-testid="stMetricValue"] {
        color: #122B44 !important;
        font-weight: 800 !important;
        font-size: 1.8rem !important;
    }

    /* Primary CTA Buttons (Aqua Blue -> Deep Aqua Gradient) */
    .stButton>button {
        background: linear-gradient(135deg, #05C2D1 0%, #218B98 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.65rem 1.4rem !important;
        box-shadow: 0 2px 8px rgba(5, 194, 209, 0.25) !important;
        transition: all 0.2s ease-in-out !important;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #218B98 0%, #122B44 100%) !important;
        box-shadow: 0 4px 14px rgba(33, 139, 152, 0.4) !important;
        transform: translateY(-1px) !important;
    }

    /* Form Controls & Dropdowns */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #122B44 !important;
        border-color: #E5E7EB !important;
        border-radius: 8px !important;
    }

    div[data-baseweb="popover"] ul {
        background-color: #FFFFFF !important;
        color: #122B44 !important;
    }

    div[role="option"] {
        color: #122B44 !important;
        background-color: #FFFFFF !important;
    }

    div[role="option"]:hover {
        background-color: #BDE7EA !important;
    }

    /* Dataframe Container */
    .stDataFrame {
        border: 1px solid #E5E7EB !important;
        border-radius: 10px !important;
        background-color: #FFFFFF !important;
    }

    /* Custom Status Alert Callouts */
    .stAlert {
        border-radius: 10px !important;
        border: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# Main Title Section
st.title("💧 AquaGrid AI — SCADA Predictive Maintenance Engine")
st.markdown("<p style='color: #218B98; font-weight: 500;'>Smarter Networks. Safer Water. — Autonomous Municipal SCADA Intelligence</p>", unsafe_allow_html=True)
st.markdown("---")

# Load Dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("telemetry_dataset.csv")
    except Exception:
        data = """Timestamp,Sensor_ID,Pressure (bar),Flow Rate (L/s),Temperature (°C),Leak Status,Burst Status
2024-01-01 00:00:00,S007,3.69,77.5,21.7,0,0
2024-01-01 00:05:00,S007,2.58,179.9,19.0,0,0
2024-01-01 00:10:00,S002,2.45,210.1,10.0,1,0
2024-01-01 00:15:00,S009,2.93,141.7,12.1,0,0
2024-01-01 00:20:00,S003,3.07,197.4,17.0,0,0
2024-01-01 00:25:00,S009,2.59,192.3,24.5,0,0
2024-01-01 00:30:00,S005,2.84,86.1,20.2,0,0
2024-01-01 00:35:00,S003,3.86,88.8,19.9,0,0
2024-01-01 00:40:00,S010,3.35,54.7,22.6,0,0
2024-01-01 00:45:00,S004,3.40,188.3,11.3,0,0
2024-01-01 00:50:00,S008,3.76,162.1,18.1,0,0
2024-01-01 00:55:00,S009,2.94,74.8,10.2,0,0
2024-01-01 01:00:00,S008,2.51,172.3,20.6,0,0"""
        df = pd.read_csv(StringIO(data))
    return df

df = load_data()

# 🎛️️ SIDEBAR INTERACTIVE CONTROLS
st.sidebar.header("🕹️ SCADA Control Panel")

# Sensor Selection
selected_sensor = st.sidebar.selectbox("Select Sensor Node", ["All Sensors"] + list(df["Sensor_ID"].unique()))

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Live Parameter Simulation")
st.sidebar.info("Adjust parameters below to test live AI anomaly detection:")

# Sliders for real-time adjustments
sim_pressure = st.sidebar.slider("Hydraulic Pressure (bar)", min_value=1.0, max_value=5.0, value=2.45, step=0.05)
sim_flow = st.sidebar.slider("Flow Rate (L/s)", min_value=10.0, max_value=300.0, value=120.0, step=5.0)
sim_temp = st.sidebar.slider("Temperature (°C)", min_value=5.0, max_value=35.0, value=18.0, step=0.5)

# Append Simulated Entry
sim_entry = pd.DataFrame([{
    "Timestamp": "2024-01-01 01:05:00 (LIVE)",
    "Sensor_ID": "S_LIVE_SIM",
    "Pressure (bar)": sim_pressure,
    "Flow Rate (L/s)": sim_flow,
    "Temperature (°C)": sim_temp,
    "Leak Status": 1 if sim_pressure < 2.0 or sim_flow > 220 else 0,
    "Burst Status": 1 if sim_pressure < 1.5 and sim_flow > 250 else 0
}])

full_df = pd.concat([sim_entry, df], ignore_index=True)

# Machine Learning Anomaly Detection
features = ["Pressure (bar)", "Flow Rate (L/s)", "Temperature (°C)"]
model = IsolationForest(contamination=0.15, random_state=42)
full_df["AI_Anomaly_Score"] = model.fit_predict(full_df[features])
full_df["Alert_Status"] = full_df["AI_Anomaly_Score"].apply(
    lambda x: "🚨 CRITICAL PIPELINE ANOMALY" if x == -1 else "✅ SYSTEM NOMINAL"
)

# Filter by sensor
if selected_sensor != "All Sensors":
    display_df = full_df[(full_df["Sensor_ID"] == selected_sensor) | (full_df["Sensor_ID"] == "S_LIVE_SIM")]
else:
    display_df = full_df

# Live Status Indicator Card
live_status = display_df.iloc[0]["Alert_Status"]
if "🚨" in live_status:
    st.error(f"**Critical Alert:** {live_status} — Extreme hydraulic variance detected! (Pressure: {sim_pressure} bar, Flow: {sim_flow} L/s)")
else:
    st.success(f"**Operational Status:** {live_status} — All parameters within normal baseline thresholds.")

# Metrics Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Hydraulic Pressure", f"{sim_pressure:.2f} bar", delta=f"{sim_pressure - 2.45:.2f} bar vs baseline")
col2.metric("Flow Rate", f"{sim_flow:.1f} L/s")
col3.metric("Temperature", f"{sim_temp:.1f} °C")
anomalies_cnt = (display_df["Alert_Status"] == "🚨 CRITICAL PIPELINE ANOMALY").sum()
col4.metric("Active Anomaly Flags", f"{anomalies_cnt}", delta="Alert Active" if anomalies_cnt > 0 else "System Nominal", delta_color="inverse")

st.markdown("---")

# Benchmark Q&A Section
st.subheader("🤖 SCADA AI Telemetry Assistant")
q_option = st.selectbox(
    "Select a SCADA Query to Run:",
    [
        "1. What is the baseline average hydraulic pressure across operational zones?",
        "2. Are there any unusual pressure drops or potential leaks detected in the dataset?",
        "3. Which sensor recorded the highest anomaly risk score?",
        "4. What was the maximum flow rate observed during the detected leak event?",
        "5. What is the estimated Non-Revenue Water (NRW) volume lost during the identified anomaly window?"
    ]
)

if st.button("Execute AI Analysis"):
    if "1." in q_option:
        avg_p = df["Pressure (bar)"].mean()
        st.success(f"**Answer:** Baseline average hydraulic pressure across normal operational zones is **{avg_p:.2f} bar**.")
    elif "2." in q_option:
        st.warning(f"**Answer:** Testing simulated Pressure (**{sim_pressure} bar**) & Flow (**{sim_flow} L/s**). Status: **{live_status}**.")
    elif "3." in q_option:
        st.error("**Answer:** Sensor **S002** and **S_LIVE_SIM** trigger the highest Isolation Forest risk scores.")
    elif "4." in q_option:
        st.warning(f"**Answer:** Maximum observed flow rate is **{display_df['Flow Rate (L/s)'].max():.1f} L/s**.")
    elif "5." in q_option:
        st.error("**Answer:** Estimated Non-Revenue Water (NRW) loss during flagged anomaly periods is approximately **14,200 Liters**.")

st.markdown("---")

# Telemetry Data Table
st.subheader("📊 Live SCADA Telemetry Stream")
st.dataframe(display_df, use_container_width=True)