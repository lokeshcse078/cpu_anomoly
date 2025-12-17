import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # REQUIRED for Streamlit Cloud
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
# Load trained Random Forest model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("cpu_rf_labeled_model.pkl")

rf = load_model()

# -----------------------------
# Sidebar UI
# -----------------------------
st.sidebar.title("Upload CPU CSV")
uploaded_file = st.sidebar.file_uploader(
    "Upload your CPU usage CSV file", type=["csv"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Required CSV format:**")
st.sidebar.code("Timestamp,cpu_usage")

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
    try:
        # -----------------------------
        # Load uploaded CSV
        # -----------------------------
        df = pd.read_csv(uploaded_file)

        # Validate columns
        if not {"Timestamp", "cpu_usage"}.issubset(df.columns):
            st.error("CSV must contain 'Timestamp' and 'cpu_usage' columns.")
            st.stop()

        # Convert timestamp safely
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df.dropna(subset=["Timestamp", "cpu_usage"], inplace=True)

        # -----------------------------
        # Feature Engineering
        # -----------------------------
        WINDOW_SIZE = 5
        df["rolling_mean"] = df["cpu_usage"].rolling(WINDOW_SIZE).mean()
        df["rolling_std"] = df["cpu_usage"].rolling(WINDOW_SIZE).std()
        df["rolling_min"] = df["cpu_usage"].rolling(WINDOW_SIZE).min()
        df["rolling_max"] = df["cpu_usage"].rolling(WINDOW_SIZE).max()
        df["delta"] = df["cpu_usage"].diff()
        df.dropna(inplace=True)

        FEATURES = [
            "cpu_usage",
            "rolling_mean",
            "rolling_std",
            "rolling_min",
            "rolling_max",
            "delta",
        ]

        # -----------------------------
        # Predict anomalies
        # -----------------------------
        df["anomaly"] = rf.predict(df[FEATURES])

        # -----------------------------
        # Plot
        # -----------------------------
        st.subheader("CPU Usage Over Time")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df["Timestamp"], df["cpu_usage"], label="CPU Usage")
        ax.scatter(
            df[df["anomaly"] == 1]["Timestamp"],
            df[df["anomaly"] == 1]["cpu_usage"],
            s=80,
            label="Detected Anomaly",
        )
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("CPU Usage (%)")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        # -----------------------------
        # Download PNG
        # -----------------------------
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        st.download_button(
            "Download Plot as PNG",
            buf,
            "cpu_anomaly_plot.png",
            "image/png",
        )

        # -----------------------------
        # Download CSV
        # -----------------------------
        st.download_button(
            "Download Predictions CSV",
            df.to_csv(index=False).encode(),
            "cpu_anomaly_predictions.csv",
            "text/csv",
        )

        # -----------------------------
        # Anomaly Table
        # -----------------------------
        st.subheader("Detected Anomalies")
        st.dataframe(df[df["anomaly"] == 1][["Timestamp", "cpu_usage"]])

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Upload a CSV file from the sidebar to see anomaly predictions.")
