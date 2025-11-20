import streamlit as st
import json
from pathlib import Path
import csv
import time
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# PAGE CONFIG & GLOBAL STYLE
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Real-Time Anomaly Detection",
    layout="wide",
)

# Global style tweaks
st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-size: 18px !important;
    }
    h1, h2, h3, h4 {
        font-size: 26px !important;
    }
    .stMetric label {
        font-size: 16px !important;
    }
    .stMetric span {
        font-size: 22px !important;
        font-weight: 600 !important;
    }
    .stButton>button {
        font-size: 18px !important;
        padding: 0.6rem 1.2rem;
        border-radius: 0.6rem;
    }
    .status-badge {
        display:inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        font-size: 14px;
        font-weight: 600;
        color: white;
    }
    .status-ok {
        background-color: #2e7d32;
    }
    .status-warn {
        background-color: #f9a825;
    }
    .status-bad {
        background-color: #c62828;
    }
    .chip {
        display:inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 999px;
        font-size: 14px;
        font-weight: 500;
        border: 1px solid #999;
        margin-right: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
anomaly_path = Path("data/current_anomaly.json")
label_log_path = Path("data/labels_log.csv")
detections_log_path = Path("data/detections_log.csv")

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def load_labels():
    if label_log_path.exists():
        try:
            df = pd.read_csv(label_log_path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def load_detections():
    if detections_log_path.exists():
        try:
            df = pd.read_csv(detections_log_path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                df = df.sort_values("timestamp")
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def compute_precision(df_labels: pd.DataFrame, col_model_flag: str):
    alerts = df_labels[df_labels[col_model_flag] == 1]
    tp = ((alerts["user_label"] == 1)).sum()
    fp = ((alerts["user_label"] == 0)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    return len(alerts), int(tp), int(fp), prec


# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------
st.sidebar.title("About this dashboard")
st.sidebar.write(
    """
    **Pipeline overview**

    1. Sensor windows stream in from Kafka.
    2. Two models score each window:
       - VAE (reconstruction error)
       - Adaptive Isolation Forest (tree ensemble)
    3. Alerts appear here for manual labeling.
    4. Your labels are used to compare model performance.
    """
)

st.sidebar.markdown("---")

detections_df_full = load_detections()
rooms = sorted(detections_df_full["resourceid"].dropna().unique()) if not detections_df_full.empty else []

selected_room = st.sidebar.selectbox(
    "Filter by room (for charts & tables)",
    options=["All rooms"] + rooms if rooms else ["All rooms"],
    index=0,
)

n_last = st.sidebar.slider(
    "Number of recent detections to visualize", min_value=50, max_value=1000, value=300, step=50
)

st.sidebar.markdown("---")
st.sidebar.info("The page auto-updates every second while streaming is active.")

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
st.title("Real-Time Anomaly Detection: VAE vs Adaptive Isolation Forest")

placeholder = st.empty()

while True:
    with placeholder.container():
        # ==============================================================
        # LOAD DATA (filtered)
        # ==============================================================
        labels_df = load_labels()
        detections_df = load_detections()

        if not detections_df.empty:
            if selected_room != "All rooms":
                detections_df = detections_df[detections_df["resourceid"] == selected_room]
            if len(detections_df) > n_last:
                detections_df = detections_df.iloc[-n_last:]

        # ==============================================================
        # 1) HIGH-LEVEL STATUS + PERFORMANCE SUMMARY
        # ==============================================================
        top1, top2 = st.columns([1.2, 2])

        with top1:
            st.subheader("Status")

            # Simple "system health" badge
            consumer_ok = not detections_df.empty
            label_count = len(labels_df) if not labels_df.empty else 0

            if consumer_ok:
                status_html = '<span class="status-badge status-ok">Streaming</span>'
            else:
                status_html = '<span class="status-badge status-warn">No detections yet</span>'

            st.markdown(status_html, unsafe_allow_html=True)
            st.write("")
            st.write(f"**Labeled events:** {label_count}")

            if selected_room != "All rooms":
                st.write(f"**Room filter:** `{selected_room}`")
            else:
                st.write("**Room filter:** All rooms")

        with top2:
            st.subheader("Model performance (from your labels)")

            if labels_df.empty:
                st.info("No labels logged yet. Label some anomalies below to see metrics.")
            else:
                expected_cols = {
                    "seq_id",
                    "resourceid",
                    "timestamp",
                    "vae_is_anomaly",
                    "if_is_anomaly",
                    "user_label",
                }
                if not expected_cols.issubset(labels_df.columns):
                    st.warning(
                        "labels_log.csv has unexpected format. "
                        "Delete it if you want to restart labeling."
                    )
                else:
                    # Optional: apply room filter to labels as well
                    labels_filtered = labels_df.copy()
                    if selected_room != "All rooms":
                        labels_filtered = labels_filtered[labels_filtered["resourceid"] == selected_room]

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.markdown("**Labels overview**")
                        st.metric("Total labeled events", len(labels_filtered))
                        st.metric(
                            "True anomalies",
                            int((labels_filtered["user_label"] == 1).sum()),
                        )
                        st.metric(
                            "False alarms",
                            int((labels_filtered["user_label"] == 0).sum()),
                        )

                    with c2:
                        st.markdown("**VAE**")
                        vae_alerts, vae_tp, vae_fp, vae_prec = compute_precision(
                            labels_filtered, "vae_is_anomaly"
                        )
                        st.metric("Alerts flagged", vae_alerts)
                        st.metric("TP / FP", f"{vae_tp} / {vae_fp}")
                        st.metric(
                            "Precision",
                            f"{vae_prec:.2f}" if vae_prec == vae_prec else "N/A",
                        )

                    with c3:
                        st.markdown("**Adaptive IF**")
                        if_alerts, if_tp, if_fp, if_prec = compute_precision(
                            labels_filtered, "if_is_anomaly"
                        )
                        st.metric("Alerts flagged", if_alerts)
                        st.metric("TP / FP", f"{if_tp} / {if_fp}")
                        st.metric(
                            "Precision",
                            f"{if_prec:.2f}" if if_prec == if_prec else "N/A",
                        )

        st.divider()

        # ==============================================================
        # 2) VISUALIZATION TABS (TIME & COMPARISON)
        # ==============================================================
        tab_scores, tab_models, tab_labels = st.tabs(
            ["📈 Scores over time", "📊 Model comparison", "📄 Label history"]
        )

        # ------------------- TAB 1: scores over time -------------------
        with tab_scores:
            st.markdown("#### Recent scores vs thresholds")

            if detections_df.empty:
                st.info("No detections logged yet. Wait for the consumer to process data.")
            else:
                st.caption(
                    f"Showing last {len(detections_df)} detections"
                    + (f" for room `{selected_room}`" if selected_room != "All rooms" else "")
                )

                if "timestamp" in detections_df.columns:
                    # VAE
                    st.markdown("**VAE reconstruction error**")
                    vae_chart_df = detections_df[["timestamp", "vae_error", "vae_threshold"]].copy()
                    vae_chart_df = vae_chart_df.set_index("timestamp")
                    st.line_chart(vae_chart_df)

                    # IF
                    st.markdown("**Adaptive IF anomaly score**")
                    if_chart_df = detections_df[["timestamp", "if_score", "if_threshold"]].copy()
                    if_chart_df = if_chart_df.set_index("timestamp")
                    st.line_chart(if_chart_df)
                else:
                    st.warning("detections_log.csv has no usable timestamp column.")

        # ---------------- TAB 2: model comparison ----------------------
        with tab_models:
            st.markdown("#### Alerts & correctness per model")

            if labels_df.empty:
                st.info("No labeled anomalies yet. Label some alerts to see this view.")
            else:
                labels_filtered = labels_df.copy()
                if selected_room != "All rooms":
                    labels_filtered = labels_filtered[labels_filtered["resourceid"] == selected_room]

                # Summary table
                summary_rows = []
                for model_name, col_flag in [("VAE", "vae_is_anomaly"), ("IF", "if_is_anomaly")]:
                    alerts = labels_filtered[labels_filtered[col_flag] == 1]
                    tp = ((alerts["user_label"] == 1)).sum()
                    fp = ((alerts["user_label"] == 0)).sum()
                    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
                    summary_rows.append(
                        {
                            "model": model_name,
                            "alerts": int(len(alerts)),
                            "TP": int(tp),
                            "FP": int(fp),
                            "precision": prec,
                        }
                    )

                summary_df = pd.DataFrame(summary_rows)

                colA, colB = st.columns(2)
                with colA:
                    st.markdown("**Alerts, TP, FP**")
                    st.dataframe(
                        summary_df[["model", "alerts", "TP", "FP"]],
                        use_container_width=True,
                    )

                with colB:
                    st.markdown("**Precision per model**")
                    if summary_df["alerts"].sum() == 0:
                        st.info("No alerts with labels yet.")
                    else:
                        chart_df = summary_df.set_index("model")[["precision"]]
                        st.bar_chart(chart_df)

                # Disagreement overview
                st.markdown("#### Where do models disagree?")

                df_disagree = labels_filtered[
                    labels_filtered["vae_is_anomaly"] != labels_filtered["if_is_anomaly"]
                ]
                st.caption(f"Disagreement cases: {len(df_disagree)}")

                if not df_disagree.empty:
                    # Show a small table of recent disagreements
                    cols = [
                        "timestamp",
                        "resourceid",
                        "vae_is_anomaly",
                        "if_is_anomaly",
                        "user_label",
                    ]
                    st.dataframe(
                        df_disagree.sort_values("timestamp").tail(50)[cols],
                        use_container_width=True,
                    )
                else:
                    st.info("So far, the models agree on all labeled events.")

        # ---------------- TAB 3: label history ------------------------
        with tab_labels:
            st.markdown("#### Labeled event history")

            if labels_df.empty:
                st.info("No labeled events yet.")
            else:
                labels_filtered = labels_df.copy()
                if selected_room != "All rooms":
                    labels_filtered = labels_filtered[labels_filtered["resourceid"] == selected_room]

                if "timestamp" in labels_filtered.columns:
                    labels_filtered = labels_filtered.sort_values("timestamp")

                show_cols = [
                    "timestamp",
                    "resourceid",
                    "vae_is_anomaly",
                    "if_is_anomaly",
                    "user_label",
                ]
                st.dataframe(
                    labels_filtered[show_cols].tail(200),
                    use_container_width=True,
                )

        st.divider()

        # ==============================================================
        # 3) LIVE ANOMALY MONITOR + LABELING
        # ==============================================================
        st.subheader("Live anomaly stream")

        if anomaly_path.exists():
            a = json.loads(anomaly_path.read_text())

            # Header row with chips
            room_text = f"`{a['resourceid']}`"
            time_text = f"`{a['timestamp']}`"

            st.markdown(
                f"**Room:** {room_text} &nbsp;&nbsp; • &nbsp;&nbsp; **Time:** {time_text}",
                unsafe_allow_html=True,
            )

            # Chips for model decisions
            chips = []
            chips.append(
                f'<span class="chip">VAE: {"ANOMALY" if a["vae_is_anomaly"] else "normal"}</span>'
            )
            chips.append(
                f'<span class="chip">IF: {"ANOMALY" if a["if_is_anomaly"] else "normal"}</span>'
            )
            if a["vae_is_anomaly"] != a["if_is_anomaly"]:
                chips.append('<span class="chip">⚠ disagreement</span>')

            st.markdown(" ".join(chips), unsafe_allow_html=True)

            colL, colR = st.columns(2)
            with colL:
                st.markdown("##### VAE")
                st.write(f"Error: `{a['vae_error']}`")
                st.write(f"Threshold: `{a['vae_threshold']}`")
                st.write(f"Is anomaly: `{a['vae_is_anomaly']}`")

            with colR:
                st.markdown("##### Adaptive Isolation Forest")
                st.write(f"Score: `{a['if_score']}`")
                st.write(f"Threshold: `{a['if_threshold']}`")
                st.write(f"Is anomaly: `{a['if_is_anomaly']}`")

            st.markdown("---")
            st.markdown("### Your judgement on this event")

            colA, colB = st.columns(2)

            # Ensure label file exists with header
            if not label_log_path.exists():
                label_log_path.parent.mkdir(parents=True, exist_ok=True)
                with label_log_path.open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "seq_id",
                            "resourceid",
                            "timestamp",
                            "vae_is_anomaly",
                            "if_is_anomaly",
                            "user_label",  # 1 = true anomaly, 0 = false alarm
                        ]
                    )

            if colA.button("✅ True anomaly", key="true_btn"):
                with label_log_path.open("a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            a["seq_id"],
                            a["resourceid"],
                            a["timestamp"],
                            int(a["vae_is_anomaly"]),
                            int(a["if_is_anomaly"]),
                            1,
                        ]
                    )
                anomaly_path.unlink(missing_ok=True)
                st.success("Logged as TRUE anomaly. Waiting for next event...")

            if colB.button("🚫 False alarm", key="false_btn"):
                with label_log_path.open("a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            a["seq_id"],
                            a["resourceid"],
                            a["timestamp"],
                            int(a["vae_is_anomaly"]),
                            int(a["if_is_anomaly"]),
                            0,
                        ]
                    )
                anomaly_path.unlink(missing_ok=True)
                st.success("Logged as FALSE alarm. Waiting for next event...")

        else:
            st.info("Monitoring... waiting for next anomaly from Kafka.")

    time.sleep(1)
    st.rerun()
