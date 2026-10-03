import streamlit as st
import numpy as np
import pandas as pd

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
# BRAND CUSTOM CSS (NAVY, AQUA BLUE, CLEAN WHITE)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Main Canvas Background */
    .stApp {
        background-color: #F8FAFC;
        color: #122B44;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headings */
    h1, h2, h3, h4 {
        color: #122B44 !important;
        font-weight: 700;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid rgba(18, 43, 68, 0.08);
    }
    
    /* Metric Cards */
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

    /* Buttons */
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
# HEADER SECTION
# ---------------------------------------------------------
st.title("💧 AquaGrid AI — SCADA Predictive Maintenance Engine")
st.caption("Smarter Networks. Safer Water. — Autonomous Municipal SCADA Intelligence")

st.markdown("---")

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.markdown("### 👤 SCADA Control Panel")
sensor_node = st.sidebar.selectbox("Select Sensor Node", ["All Sensors", "Node-Zone-01", "Node-Zone-04B", "Node-Zone-09"])

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Live Parameter Simulation")
st.sidebar.info("Adjust parameters below to test live AI anomaly detection:")

# Live Parameter Sliders
pressure = st.sidebar.slider("Hydraulic Pressure (bar)", min_value=0.0, max_value=6.0, value=2.45, step=0.05)
flow_rate = st.sidebar.slider("Flow Rate (L/s)", min_value=0.0, max_value=250.0, value=120.0, step=1.0)
temperature = st.sidebar.slider("Temperature (°C)", min_value=0.0, max_value=40.0, value=18.0, step=0.5)
acoustic_noise = st.sidebar.slider("Acoustic Noise (dB)", min_value=20.0, max_value=110.0, value=42.0, step=1.0)

# ---------------------------------------------------------
# AI ISOLATION FOREST ANOMALY LOGIC (SIMULATED MODEL)
# ---------------------------------------------------------
# Baseline defaults: Pressure ~ 2.45 bar, Flow ~ 120 L/s, Temp ~ 18 C, Acoustic ~ 40 dB
pressure_dev = abs(pressure - 2.45) / 2.45
flow_dev = abs(flow_rate - 120.0) / 120.0
temp_dev = abs(temperature - 18.0) / 18.0
acoustic_dev = max(0.0, (acoustic_noise - 45.0) / 45.0)

# Calculate composite Isolation Forest anomaly probability score
composite_score = (pressure_dev * 0.35) + (flow_dev * 0.25) + (temp_dev * 0.15) + (acoustic_dev * 0.25)
anomaly_prob = min(0.998, max(0.021, composite_score))

# Status Determination
if anomaly_prob > 0.60:
    op_status = "⚠️ CRITICAL ANOMALY DETECTED — Pressure or acoustic threshold breached."
    status_box_type = st.error
    anomaly_flag_count = int(150 + (anomaly_prob * 300))
elif anomaly_prob > 0.30:
    op_status = "⚡ WARNING — Mild operational variance detected."
    status_box_type = st.warning
    anomaly_flag_count = int(50 + (anomaly_prob * 100))
else:
    op_status = "✅ SYSTEM NOMINAL — All parameters within normal baseline thresholds."
    status_box_type = st.success
    anomaly_flag_count = 150

# Display Operational Banner
status_box_type(f"**Operational Status:** {op_status}")

# ---------------------------------------------------------
# MAIN METRIC CARDS (INCLUDES ACOUSTIC & ANOMALY PROBABILITY)
# ---------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    pressure_diff = pressure - 2.45
    st.metric(
        label="Hydraulic Pressure",
        value=f"{pressure:.2f} bar",
        delta=f"{pressure_diff:+.2f} bar vs baseline"
    )

with col2:
    st.metric(
        label="Flow Rate",
        value=f"{flow_rate:.1f} L/s"
    )

with col3:
    st.metric(
        label="Temperature",
        value=f"{temperature:.1f} °C"
    )

with col4:
    st.metric(
        label="Acoustic Noise",
        value=f"{acoustic_noise:.0f} dB",
        delta="High Noise" if acoustic_noise > 65 else "Normal",
        delta_color="inverse" if acoustic_noise > 65 else "normal"
    )

with col5:
    st.metric(
        label="AI Anomaly Prob.",
        value=f"{anomaly_prob * 100:.1f}%",
        delta="ALERT ACTIVE" if anomaly_prob > 0.30 else "NOMINAL",
        delta_color="inverse" if anomaly_prob > 0.30 else "normal"
    )

st.markdown("---")

# ---------------------------------------------------------
# INTERACTIVE SCADA AI TELEMETRY ASSISTANT
# ---------------------------------------------------------
st.subheader("🤖 SCADA AI Telemetry Assistant")

query_option = st.selectbox(
    "Select a SCADA Query to Run:",
    [
        "1. What is the baseline average hydraulic pressure across operational zones?",
        "2. Is there an active pressure or acoustic anomaly detected in the network?",
        "3. What is the estimated water loss rate and financial risk (NRW)?",
        "4. Which pump/valve asset requires immediate preventive maintenance?",
        "5. Generate an automated dispatch ticket for field engineers."
    ]
)

if st.button("Execute AI Analysis"):
    st.markdown("### 📋 AI Diagnostic Output")
    
    if "1." in query_option:
        st.write(
            f"**System Baseline Report:** Nominal hydraulic pressure across all operational zones is **2.45 bar**. "
            f"Current measured pressure is **{pressure:.2f} bar** (Variance: {pressure_diff:+.2f} bar)."
        )
        
    elif "2." in query_option:
        st.write(
            f"**Isolation Forest Anomaly Analysis:**\n"
            f"- **Calculated Anomaly Probability:** `{anomaly_prob * 100:.1f}%`\n"
            f"- **Acoustic Leak Signature:** `{acoustic_noise:.0f} dB`\n"
            f"- **Status:** {'CRITICAL SENSOR SPIKE DETECTED' if anomaly_prob > 0.30 else 'No active critical leaks detected.'}"
        )
        
    elif "3." in query_option:
        estimated_loss_lps = round(anomaly_prob * 18.5, 2)
        financial_loss = int(estimated_loss_lps * 3000)
        st.write(
            f"**Non-Revenue Water (NRW) Impact Analysis:**\n"
            f"- **Estimated Water Loss:** `{estimated_loss_lps} Liters/sec`\n"
            f"- **Projected Revenue Loss:** `₹{financial_loss:,} / day`"
        )
        
    elif "4." in query_option:
        st.write(
            f"**Predictive Asset Health Score:**\n"
            f"- **Target Asset:** Pump #PUMP-02 (Zone 04B)\n"
            f"- **Vibration / Cavitation Risk:** `{acoustic_noise:.0f} dB`\n"
            f"- **Remaining Useful Life (RUL):** `{'48 Hours' if acoustic_noise > 60 else '320 Hours'}`"
        )
        
    elif "5." in query_option:
        st.success(
            f"**AUTOMATED WORK ORDER GENERATED (#WO-8921)**\n\n"
            f"- **Location:** Zone 04B Junction 12\n"
            f"- **Trigger:** Anomaly Probability {anomaly_prob * 100:.1f}% | Acoustic Noise {acoustic_noise:.0f} dB\n"
            f"- **Assigned Team:** Crew B (Field Maintenance)\n"
            f"- **Recommended Action:** Inspect main valve seal and reduce line pressure transient."
        )

# Footer
st.markdown("---")
st.caption("AquaGrid AI Platform • Connected to SCADA Node Simulator • Streamlit v1.38+")