import os
import json
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

# ---------------------------------------------------------------------
# PAGE CONFIG & GLOBAL STYLE
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Real-Time Anomaly Detection (Adaptive IF)",
    layout="wide",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"]  { font-size: 18px !important; }
    h1, h2, h3, h4 { font-size: 26px !important; }
    .stMetric label { font-size: 16px !important; }
    .stMetric span { font-size: 22px !important; font-weight: 600 !important; }
    .stButton>button { font-size: 18px !important; padding: 0.6rem 1.2rem; border-radius: 0.6rem; }
    .status-badge {
        display:inline-block; padding: 0.2rem 0.6rem; border-radius: 999px;
        font-size: 14px; font-weight: 600; color: white;
    }
    .status-ok { background-color: #2e7d32; }
    .status-warn { background-color: #f9a825; }
    .chip {
        display:inline-block; padding: 0.15rem 0.5rem; border-radius: 999px;
        font-size: 14px; font-weight: 500; border: 1px solid #999; margin-right: 0.25rem;
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
def load_labels() -> pd.DataFrame:
    if not label_log_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(label_log_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df
    except Exception as e:
        st.warning(f"Failed to read labels_log.csv: {e}")
        return pd.DataFrame()


def load_detections() -> pd.DataFrame:
    """
    Robust loading:
    - shows file path in UI
    - if tokenizer error happens, tries python engine
    """
    st.caption(f"DEBUG CWD: {os.getcwd()}")
    st.caption(f"DEBUG detections path: {detections_log_path.resolve()}")

    if not detections_log_path.exists():
        st.warning("detections_log.csv not found at this path.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(detections_log_path)
    except Exception as e1:
        st.warning(f"Failed to read detections_log.csv (default engine): {e1}")
        try:
            df = pd.read_csv(detections_log_path, engine="python")
            st.info("Recovered detections_log.csv using engine='python'.")
        except Exception as e2:
            st.warning(f"Failed to read detections_log.csv (python engine): {e2}")
            return pd.DataFrame()

    st.caption(f"DEBUG detections raw rows: {len(df)}, cols: {list(df.columns)}")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp")

    # Normalize seq_id type for safer merges
    if "seq_id" in df.columns:
        df["seq_id"] = df["seq_id"].astype(str).str.strip()

    return df


def compute_precision(labels_df: pd.DataFrame):
    if labels_df.empty or "if_is_anomaly" not in labels_df.columns or "user_label" not in labels_df.columns:
        return 0, 0, 0, float("nan")

    alerts = labels_df[labels_df["if_is_anomaly"].astype(int) == 1]
    tp = int((alerts["user_label"].astype(int) == 1).sum())
    fp = int((alerts["user_label"].astype(int) == 0).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    return int(len(alerts)), tp, fp, prec


def prepare_eval_dataframe(selected_room: str) -> pd.DataFrame:
    det = load_detections()
    lab = load_labels()

    if det.empty or lab.empty:
        return pd.DataFrame()

    # normalize seq_id for merge
    if "seq_id" in lab.columns:
        lab["seq_id"] = lab["seq_id"].astype(str).str.strip()

    cols_det = [c for c in ["seq_id", "resourceid", "if_score"] if c in det.columns]
    cols_lab = [c for c in ["seq_id", "resourceid", "user_label"] if c in lab.columns]

    if len(cols_det) < 2 or len(cols_lab) < 2:
        return pd.DataFrame()

    det_small = det[cols_det].copy()
    lab_small = lab[cols_lab].copy()

    df = pd.merge(det_small, lab_small, on="seq_id", how="inner", suffixes=("_det", "_lab"))

    # normalize resourceid
    if "resourceid_det" in df.columns:
        df["resourceid"] = df["resourceid_det"]
    elif "resourceid_lab" in df.columns:
        df["resourceid"] = df["resourceid_lab"]

    if selected_room != "All rooms" and "resourceid" in df.columns:
        df = df[df["resourceid"] == selected_room]

    if "user_label" not in df.columns or "if_score" not in df.columns:
        return pd.DataFrame()

    df = df.dropna(subset=["user_label", "if_score"])
    return df


def _clean_scores(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    mask = np.isfinite(scores)
    y_clean = y_true[mask]
    s_clean = scores[mask]

    if len(s_clean) < 2 or len(np.unique(y_clean)) < 2:
        return None, None

    s_clean = np.clip(s_clean, -1e9, 1e9)
    return y_clean, s_clean


def compute_roc_pr(y_true, scores):
    y_clean, s_clean = _clean_scores(y_true, scores)
    if y_clean is None:
        return None

    fpr, tpr, _ = roc_curve(y_clean, s_clean)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_clean, s_clean)
    ap = average_precision_score(y_clean, s_clean)

    return {"fpr": fpr, "tpr": tpr, "roc_auc": roc_auc, "precision": precision, "recall": recall, "ap": ap}


def best_f1_from_pr(y_true, scores):
    y_clean, s_clean = _clean_scores(y_true, scores)
    if y_clean is None:
        return float("nan")

    precision, recall, _ = precision_recall_curve(y_clean, s_clean)
    denom = precision + recall
    denom[denom == 0] = 1e-9
    f1 = 2 * precision * recall / denom
    return float(np.nanmax(f1)) if len(f1) else float("nan")


def build_confirmed_points(det_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      confirmed_true_anomaly = 1 if user_label==1 for that seq_id
      confirmed_false_alarm  = 1 if user_label==0 for that seq_id
    seq_id is normalized as string to avoid merge mismatches.
    """
    out = det_df.copy()
    out["confirmed_true_anomaly"] = 0
    out["confirmed_false_alarm"] = 0

    if out.empty or labels_df.empty:
        return out

    if "seq_id" not in out.columns or "seq_id" not in labels_df.columns:
        return out

    out["seq_id"] = out["seq_id"].astype(str).str.strip()

    lab = labels_df.copy()
    if "seq_id" not in lab.columns or "user_label" not in lab.columns:
        return out

    lab = lab[["seq_id", "user_label"]].dropna(subset=["seq_id", "user_label"])
    lab["seq_id"] = lab["seq_id"].astype(str).str.strip()
    lab["user_label"] = lab["user_label"].astype(int)

    lab = lab.drop_duplicates(subset=["seq_id"], keep="last")
    out = out.merge(lab, on="seq_id", how="left")

    out["confirmed_true_anomaly"] = (out["user_label"] == 1).astype(int)
    out["confirmed_false_alarm"] = (out["user_label"] == 0).astype(int)
    return out


# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------
st.sidebar.title("About this dashboard")
st.sidebar.write(
    """
**Pipeline overview**
1. Sensor windows stream in from Kafka (`nimway-sensors`).
2. Adaptive Isolation Forest produces `if_score` + `if_is_anomaly`.
3. Alerts appear for manual labeling.
4. Your labels evaluate the model (ROC/PR/F1).
"""
)
st.sidebar.markdown("---")

detections_df_full = load_detections()

if not detections_df_full.empty and "resourceid" in detections_df_full.columns:
    rooms = sorted(detections_df_full["resourceid"].dropna().astype(str).unique())
else:
    rooms = []

selected_room = st.sidebar.selectbox(
    "Filter by room",
    options=["All rooms"] + rooms if rooms else ["All rooms"],
    index=0,
)

n_last = st.sidebar.slider(
    "Number of recent detections to visualize",
    min_value=50,
    max_value=2000,
    value=300,
    step=50,
)

st.sidebar.markdown("---")
st.sidebar.info("Auto-updates every 1 second while streaming is active.")

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
st.title("Real-Time Anomaly Detection – Adaptive Isolation Forest")

placeholder = st.empty()

while True:
    with placeholder.container():
        labels_df = load_labels()
        detections_df = load_detections()

        # Apply same room filter to labels for consistency
        if selected_room != "All rooms" and not labels_df.empty and "resourceid" in labels_df.columns:
            labels_df = labels_df[labels_df["resourceid"] == selected_room]

        # Filters for detections
        if not detections_df.empty:
            if selected_room != "All rooms" and "resourceid" in detections_df.columns:
                detections_df = detections_df[detections_df["resourceid"] == selected_room]

            if len(detections_df) > n_last:
                detections_df = detections_df.iloc[-n_last:]

        # =================== STATUS & METRICS ==========================
        top1, top2 = st.columns([1.2, 2])

        with top1:
            st.subheader("Status")
            consumer_ok = not detections_df.empty
            label_count = len(labels_df) if not labels_df.empty else 0

            st.markdown(
                '<span class="status-badge status-ok">Streaming</span>' if consumer_ok
                else '<span class="status-badge status-warn">No detections yet</span>',
                unsafe_allow_html=True,
            )
            st.write("")
            st.write(f"**Labeled events:** {label_count}")
            st.write(f"**Room filter:** `{selected_room}`" if selected_room != "All rooms" else "**Room filter:** All rooms")

        with top2:
            st.subheader("Model performance (from your labels)")
            if labels_df.empty:
                st.info("No labels yet. Label some events below to see metrics.")
            else:
                alerts, tp, fp, prec = compute_precision(labels_df)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Alerts flagged", alerts)
                c2.metric("True positives", tp)
                c3.metric("False positives", fp)
                c4.metric("Precision", f"{prec:.2f}" if prec == prec else "N/A")

        st.divider()

        # ===================== TABS ====================================
        tab_scores, tab_labels_tab, tab_eval = st.tabs(
            ["📈 Scores over time", "📄 Label history", "📐 ROC & PR evaluation"]
        )

        # ----- TAB: scores over time -----
        with tab_scores:
            st.markdown("#### IF Score vs Threshold")
            st.caption("🟡 Yellow = machine anomaly (unconfirmed) • 🔴 Red = confirmed TRUE • ⚪ Gray = confirmed FALSE")

            if detections_df.empty:
                st.info("No detections available after filters. Wait for new data or change room.")
            else:
                det_plot = build_confirmed_points(detections_df, labels_df)

                # Debug counts
                if "confirmed_true_anomaly" in det_plot.columns:
                    st.caption(f"DEBUG confirmed TRUE in view: {int(det_plot['confirmed_true_anomaly'].sum())}")
                if "if_is_anomaly" in det_plot.columns:
                    st.caption(f"DEBUG machine anomalies in view: {int((det_plot['if_is_anomaly'].astype(float).fillna(0).astype(int)==1).sum())}")

                # ensure timestamp exists
                if "timestamp" not in det_plot.columns:
                    st.warning("detections_log.csv missing timestamp column.")
                else:
                    det_plot = det_plot.dropna(subset=["timestamp"]).copy()
                    det_plot = det_plot.sort_values("timestamp")

                    # ensure numeric columns
                    if "if_score" in det_plot.columns:
                        det_plot["if_score"] = pd.to_numeric(det_plot["if_score"], errors="coerce")
                    if "if_threshold" in det_plot.columns:
                        det_plot["if_threshold"] = pd.to_numeric(det_plot["if_threshold"], errors="coerce")
                    if "if_is_anomaly" in det_plot.columns:
                        det_plot["if_is_anomaly"] = pd.to_numeric(det_plot["if_is_anomaly"], errors="coerce").fillna(0).astype(int)

                    fig = go.Figure()

                    # Line: score
                    fig.add_trace(
                        go.Scatter(
                            x=det_plot["timestamp"],
                            y=det_plot["if_score"],
                            mode="lines",
                            name="IF score",
                        )
                    )

                    # Dashed threshold if present
                    if "if_threshold" in det_plot.columns and det_plot["if_threshold"].notna().any():
                        fig.add_trace(
                            go.Scatter(
                                x=det_plot["timestamp"],
                                y=det_plot["if_threshold"],
                                mode="lines",
                                name="Threshold",
                                line=dict(dash="dash"),
                            )
                        )

                    # confirmed
                    confirmed_true = det_plot[det_plot["confirmed_true_anomaly"] == 1]
                    confirmed_false = det_plot[det_plot["confirmed_false_alarm"] == 1]

                    # machine-only anomalies (yellow)
                    if "if_is_anomaly" in det_plot.columns:
                        machine_anoms = det_plot[det_plot["if_is_anomaly"] == 1]
                        machine_only = machine_anoms[
                            (machine_anoms["confirmed_true_anomaly"] == 0)
                            & (machine_anoms["confirmed_false_alarm"] == 0)
                        ]
                    else:
                        machine_only = det_plot.iloc[0:0]

                    if not machine_only.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=machine_only["timestamp"],
                                y=machine_only["if_score"],
                                mode="markers",
                                name="Machine anomaly (unconfirmed)",
                                marker=dict(color="gold", size=9),
                                hovertemplate="time=%{x}<br>score=%{y}<extra></extra>",
                            )
                        )

                    if not confirmed_true.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=confirmed_true["timestamp"],
                                y=confirmed_true["if_score"],
                                mode="markers",
                                name="Confirmed TRUE anomaly",
                                marker=dict(color="red", size=10),
                                hovertemplate="time=%{x}<br>score=%{y}<extra></extra>",
                            )
                        )

                    if not confirmed_false.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=confirmed_false["timestamp"],
                                y=confirmed_false["if_score"],
                                mode="markers",
                                name="Confirmed FALSE alarm",
                                marker=dict(color="gray", size=8, symbol="circle-open"),
                                hovertemplate="time=%{x}<br>score=%{y}<extra></extra>",
                            )
                        )

                    fig.update_layout(
                        height=460,
                        xaxis_title="Time",
                        yaxis_title="IF score",
                        legend_title_text="Legend",
                        margin=dict(l=10, r=10, t=35, b=10),
                    )

                    st.plotly_chart(fig, use_container_width=True)

        # ----- TAB: label history -----
        with tab_labels_tab:
            st.markdown("#### Labeled event history")

            if labels_df.empty:
                st.info("No labeled events yet.")
            else:
                if "timestamp" in labels_df.columns:
                    labels_df = labels_df.sort_values("timestamp")

                show_cols = [c for c in ["timestamp", "resourceid", "if_is_anomaly", "user_label", "seq_id"] if c in labels_df.columns]
                st.dataframe(labels_df[show_cols].tail(200), use_container_width=True)

        # ----- TAB: ROC & PR evaluation -----
        with tab_eval:
            st.markdown("#### ROC & Precision–Recall evaluation (Adaptive IF)")

            df_eval = prepare_eval_dataframe(selected_room)

            if df_eval.empty:
                st.info("Not enough labeled data yet. Label more events in the live stream section.")
            else:
                y_true = df_eval["user_label"].astype(int).values
                s_if = pd.to_numeric(df_eval["if_score"], errors="coerce").values

                res_if = compute_roc_pr(y_true, s_if)

                if res_if is None:
                    st.info("Need both normal and anomalous labels to compute ROC/PR curves.")
                else:
                    best_f1_if = best_f1_from_pr(y_true, s_if)

                    eval_row = {
                        "AUC-ROC": res_if["roc_auc"],
                        "AUC-PR": res_if["ap"],
                        "Best F1 (approx)": best_f1_if,
                    }
                    st.dataframe(pd.DataFrame([eval_row], index=["Adaptive IF"]), use_container_width=True)

                    colR1, colR2 = st.columns(2)

                    with colR1:
                        st.markdown("**ROC curve**")
                        roc_df = pd.DataFrame({"FPR": res_if["fpr"], "TPR": res_if["tpr"]}).set_index("FPR")
                        st.line_chart(roc_df)

                    with colR2:
                        st.markdown("**Precision–Recall curve**")
                        pr_df = pd.DataFrame({"Recall": res_if["recall"], "Precision": res_if["precision"]}).set_index("Recall")
                        st.line_chart(pr_df)

        st.divider()

        # ===================== LIVE ANOMALY STREAM =====================
        st.subheader("Live anomaly stream (Adaptive IF)")

        if anomaly_path.exists():
            a = json.loads(anomaly_path.read_text())

            st.markdown(
                f"**Room:** `{a.get('resourceid','')}` &nbsp;&nbsp; • &nbsp;&nbsp; **Time:** `{a.get('timestamp','')}`",
                unsafe_allow_html=True,
            )

            chips = [f'<span class="chip">IF: {"ANOMALY" if a.get("if_is_anomaly") else "normal"}</span>']
            st.markdown(" ".join(chips), unsafe_allow_html=True)

            st.markdown("##### Adaptive Isolation Forest")
            st.write(f"Score: `{a.get('if_score')}`")
            st.write(f"Threshold: `{a.get('if_threshold')}`")
            st.write(f"Is anomaly: `{a.get('if_is_anomaly')}`")

            st.markdown("---")
            st.markdown("### Your judgement on this event")

            colA, colB = st.columns(2)

            # Ensure label file exists with header
            if not label_log_path.exists():
                label_log_path.parent.mkdir(parents=True, exist_ok=True)
                with label_log_path.open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["seq_id", "resourceid", "timestamp", "if_is_anomaly", "user_label"])

            if colA.button("✅ True anomaly", key="true_btn"):
                with label_log_path.open("a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([a.get("seq_id"), a.get("resourceid"), a.get("timestamp"), int(bool(a.get("if_is_anomaly"))), 1])
                anomaly_path.unlink(missing_ok=True)
                st.success("Logged as TRUE anomaly. Waiting for next event...")

            if colB.button("🚫 False alarm", key="false_btn"):
                with label_log_path.open("a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([a.get("seq_id"), a.get("resourceid"), a.get("timestamp"), int(bool(a.get("if_is_anomaly"))), 0])
                anomaly_path.unlink(missing_ok=True)
                st.success("Logged as FALSE alarm. Waiting for next event...")

        else:
            st.info("Monitoring... waiting for next anomaly from Kafka.")

    time.sleep(1)
    st.rerun()
