import streamlit as st
import json
from pathlib import Path
import csv

st.title("Real-Time Anomaly Detection (VAE vs IF)")

anomaly_path = Path("data/current_anomaly.json")
label_log = Path("data/labels_log.csv")
if not label_log.exists():
    with label_log.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seq_id", "resourceid", "timestamp",
            "vae_is_anomaly", "if_is_anomaly",
            "user_label",  # 1=real anomaly, 0=false alarm
        ])

if anomaly_path.exists():
    a = json.loads(anomaly_path.read_text())
    st.write(f"Room {a['resourceid']} @ {a['timestamp']}")
    st.write(f"VAE: error {a['vae_error']} > {a['vae_threshold']} ? {a['vae_is_anomaly']}")
    st.write(f"IF:  score {a['if_score']} > {a['if_threshold']} ? {a['if_is_anomaly']}")

    col1, col2 = st.columns(2)
    if col1.button("False Alarm"):
        label = 0
        with label_log.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                a["seq_id"], a["resourceid"], a["timestamp"],
                a["vae_is_anomaly"], a["if_is_anomaly"], label,
            ])
        anomaly_path.unlink()
        st.success("Logged: False alarm")

    if col2.button("True Anomaly"):
        label = 1
        with label_log.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                a["seq_id"], a["resourceid"], a["timestamp"],
                a["vae_is_anomaly"], a["if_is_anomaly"], label,
            ])
        anomaly_path.unlink()
        st.success("Logged: True anomaly")

else:
    st.info("Monitoring...")
