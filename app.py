import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from io import BytesIO

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="CPU Anomaly Detection",
    page_icon="🖥️",
    layout="wide",
)

# -----------------------------
# Load trained Random Forest model
# -----------------------------
rf = joblib.load("cpu_rf_labeled_model.pkl")

# -----------------------------
# Sidebar UI
# -----------------------------
st.sidebar.title("Upload CPU CSV")
uploaded_file = st.sidebar.file_uploader(
    "Upload your CPU usage CSV file", type=["csv"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("**CSV format:** Timestamp,cpu_usage")

# -----------------------------
# Main container
# -----------------------------
st.title("🖥️ CPU Anomaly Detection Dashboard")
st.markdown(
    """
Detect CPU anomalies using **Random Forest trained on labeled data**.
Red points indicate detected anomalies.
"""
)

if uploaded_file:
    # -----------------------------
    # Load uploaded CSV
    # -----------------------------
    df = pd.read_csv(uploaded_file)
    
    # Convert timestamp to datetime
    if df['Timestamp'].dtype != 'datetime64[ns]':
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # -----------------------------
    # Feature Engineering
    # -----------------------------
    WINDOW_SIZE = 5
    df['rolling_mean'] = df['cpu_usage'].rolling(WINDOW_SIZE).mean()
    df['rolling_std'] = df['cpu_usage'].rolling(WINDOW_SIZE).std()
    df['rolling_min'] = df['cpu_usage'].rolling(WINDOW_SIZE).min()
    df['rolling_max'] = df['cpu_usage'].rolling(WINDOW_SIZE).max()
    df['delta'] = df['cpu_usage'].diff()
    df.dropna(inplace=True)

    FEATURES = ['cpu_usage','rolling_mean','rolling_std','rolling_min','rolling_max','delta']

    # -----------------------------
    # Predict anomalies
    # -----------------------------
    df['anomaly'] = rf.predict(df[FEATURES])

    # -----------------------------
    # Interactive plot
    # -----------------------------
    st.subheader("CPU Usage Over Time")
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(df['Timestamp'], df['cpu_usage'], color='lightgray', label='CPU Usage', zorder=1)
    ax.scatter(df['Timestamp'][df['anomaly']==1], df['cpu_usage'][df['anomaly']==1], 
               color='red', s=80, label='Detected Anomaly', zorder=5)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("CPU Usage (%)")
    ax.set_title("CPU Usage with Random Forest Anomaly Detection")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # -----------------------------
    # Downloadable PNG
    # -----------------------------
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    st.download_button(
        label="Download Plot as PNG",
        data=buf,
        file_name="cpu_anomaly_plot.png",
        mime="image/png"
    )

    # -----------------------------
    # Download predictions CSV
    # -----------------------------
    csv_buffer = df.to_csv(index=False).encode()
    st.download_button(
        label="Download Predictions CSV",
        data=csv_buffer,
        file_name="cpu_anomaly_predictions.csv",
        mime="text/csv"
    )

    # -----------------------------
    # Show table of anomalies
    # -----------------------------
    st.subheader("Detected Anomalies")
    st.dataframe(df[df['anomaly']==1][['Timestamp','cpu_usage']])

else:
    st.info("Upload a CSV file from the sidebar to see anomaly predictions.")
