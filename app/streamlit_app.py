import os
import json
import csv
import time
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

# ---------------------------------------------------------------------
# PAGE CONFIG & GLOBAL STYLE
# ---------------------------------------------------------------------
st.set_page_config(page_title="Real-Time Anomaly Detection (Adaptive IF)", layout="wide")

st.markdown(
    """
    <style>
    html, body, [class*="css"]  { font-size: 18px !important; }
    h1, h2, h3, h4 { font-size: 26px !important; }

    .stMetric label { font-size: 16px !important; }
    .stMetric span  { font-size: 22px !important; font-weight: 650 !important; }

    .stButton>button { font-size: 18px !important; padding: 0.65rem 1.2rem; border-radius: 0.65rem; }

    .status-badge {
        display:inline-block; padding: 0.25rem 0.7rem; border-radius: 999px;
        font-size: 14px; font-weight: 700; color: white;
    }
    .status-ok   { background-color: #2e7d32; }
    .status-warn { background-color: #f9a825; }
    .status-bad  { background-color: #c62828; }

    .chip {
        display:inline-block; padding: 0.2rem 0.55rem; border-radius: 999px;
        font-size: 14px; font-weight: 600; border: 1px solid #999; margin-right: 0.35rem;
    }

    .lock-card {
        border: 2px solid #f9a825;
        border-radius: 16px;
        padding: 14px 16px;
        background: rgba(249,168,37,0.10);
        margin-bottom: 12px;
    }
    .lock-title {
        font-size: 18px;
        font-weight: 800;
        color: #7a4f00;
        margin: 0;
    }
    .lock-sub {
        font-size: 14px;
        margin-top: 6px;
        color: #7a4f00;
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
# SESSION STATE
# ---------------------------------------------------------------------
if "review_lock" not in st.session_state:
    st.session_state.review_lock = None  # stores anomaly dict currently under review (frozen)

if "snooze_until" not in st.session_state:
    st.session_state.snooze_until = 0.0  # timestamp, if > now hide alert temporarily

# ---------------------------------------------------------------------
# HELPERS: IO
# ---------------------------------------------------------------------
def load_labels() -> pd.DataFrame:
    if not label_log_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(label_log_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if "seq_id" in df.columns:
            df["seq_id"] = df["seq_id"].astype(str).str.strip()
        return df
    except Exception as e:
        st.warning(f"Failed to read labels_log.csv: {e}")
        return pd.DataFrame()


def load_detections() -> pd.DataFrame:
    # Comment out debug captions when stable:
    st.caption(f"DEBUG CWD: {os.getcwd()}")
    st.caption(f"DEBUG detections path: {detections_log_path.resolve()}")

    if not detections_log_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(detections_log_path)
    except Exception:
        try:
            df = pd.read_csv(detections_log_path, engine="python", on_bad_lines="skip")
            st.info("Recovered detections_log.csv using python engine (skipping bad lines).")
        except Exception as e2:
            st.warning(f"Failed to read detections_log.csv: {e2}")
            return pd.DataFrame()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp")

    if "seq_id" in df.columns:
        df["seq_id"] = df["seq_id"].astype(str).str.strip()

    return df


def read_current_anomaly_json() -> Optional[Dict]:
    if not anomaly_path.exists():
        return None
    try:
        return json.loads(anomaly_path.read_text())
    except Exception as e:
        st.warning(f"Failed to parse current_anomaly.json: {e}")
        return None


def ensure_labels_file():
    if label_log_path.exists():
        return
    label_log_path.parent.mkdir(parents=True, exist_ok=True)
    with label_log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seq_id", "resourceid", "timestamp", "if_is_anomaly", "user_label"])


# ---------------------------------------------------------------------
# HELPERS: metrics
# ---------------------------------------------------------------------
def compute_precision(labels_df: pd.DataFrame):
    if labels_df.empty:
        return 0, 0, 0, float("nan")
    if "if_is_anomaly" not in labels_df.columns or "user_label" not in labels_df.columns:
        return 0, 0, 0, float("nan")

    alerts = labels_df[labels_df["if_is_anomaly"].astype(int) == 1]
    tp = int((alerts["user_label"].astype(int) == 1).sum())
    fp = int((alerts["user_label"].astype(int) == 0).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    return int(len(alerts)), tp, fp, prec


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


def prepare_eval_dataframe(selected_room: str) -> pd.DataFrame:
    det = load_detections()
    lab = load_labels()
    if det.empty or lab.empty:
        return pd.DataFrame()

    cols_det = [c for c in ["seq_id", "resourceid", "if_score"] if c in det.columns]
    cols_lab = [c for c in ["seq_id", "resourceid", "user_label"] if c in lab.columns]
    if len(cols_det) < 2 or len(cols_lab) < 2:
        return pd.DataFrame()

    det_small = det[cols_det].copy()
    lab_small = lab[cols_lab].copy()

    df = pd.merge(det_small, lab_small, on="seq_id", how="inner", suffixes=("_det", "_lab"))

    if "resourceid_det" in df.columns:
        df["resourceid"] = df["resourceid_det"]
    elif "resourceid_lab" in df.columns:
        df["resourceid"] = df["resourceid_lab"]

    if selected_room != "All rooms" and "resourceid" in df.columns:
        df = df[df["resourceid"] == selected_room]

    df = df.dropna(subset=["user_label", "if_score"])
    return df


def build_confirmed_points(det_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    out = det_df.copy()
    out["confirmed_true_anomaly"] = 0
    out["confirmed_false_alarm"] = 0

    if out.empty or labels_df.empty:
        return out
    if "seq_id" not in out.columns or "seq_id" not in labels_df.columns or "user_label" not in labels_df.columns:
        return out

    out["seq_id"] = out["seq_id"].astype(str).str.strip()
    lab = labels_df[["seq_id", "user_label"]].dropna()
    lab["seq_id"] = lab["seq_id"].astype(str).str.strip()
    lab["user_label"] = lab["user_label"].astype(int)
    lab = lab.drop_duplicates(subset=["seq_id"], keep="last")

    out = out.merge(lab, on="seq_id", how="left")

    out["confirmed_true_anomaly"] = (out["user_label"] == 1).astype(int)
    out["confirmed_false_alarm"] = (out["user_label"] == 0).astype(int)
    return out


# ---------------------------------------------------------------------
# HELPERS: anomaly details for UI
# ---------------------------------------------------------------------
SENSOR_BASE = [
    "temperature_value",
    "humidity_value",
    "co2_value",
    "light_value",
    "radon_value",
    "airquality_value",
]


def extract_sensor_snapshot(a: dict) -> pd.DataFrame:
    rows = []
    for s in SENSOR_BASE:
        k_mean = f"{s}_mean"
        k_min = f"{s}_min"
        k_max = f"{s}_max"
        if k_mean in a or k_min in a or k_max in a:
            rows.append({"sensor": s, "mean": a.get(k_mean), "min": a.get(k_min), "max": a.get(k_max)})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def extract_shap_importance(a: dict) -> pd.DataFrame:
    shap_dict = a.get("if_shap_importance")
    if not isinstance(shap_dict, dict) or not shap_dict:
        return pd.DataFrame()
    return (
        pd.DataFrame({"sensor": list(shap_dict.keys()), "importance": list(shap_dict.values())})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def plot_shap_bar(df_shap: pd.DataFrame, top_k: int = 6):
    dfp = df_shap.head(top_k).copy()
    if dfp.empty:
        return

    colors = ["red"] + ["#f9a825"] * (len(dfp) - 1)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=dfp["importance"],
            y=dfp["sensor"],
            orientation="h",
            marker=dict(color=colors),
            hovertemplate="sensor=%{y}<br>importance=%{x}<extra></extra>",
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Mean |SHAP| (higher = stronger driver)",
        yaxis_title="Sensor",
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_sensor_snapshot_bar(snap: pd.DataFrame, shap: pd.DataFrame):
    """
    Bar plot of mean with min/max error bars.
    Coloring:
      - if SHAP exists, highlight top sensor red, others yellow
      - else highlight largest range red, others yellow
    """
    if snap.empty:
        return

    df = snap.copy()
    df["mean"] = pd.to_numeric(df["mean"], errors="coerce")
    df["min"] = pd.to_numeric(df["min"], errors="coerce")
    df["max"] = pd.to_numeric(df["max"], errors="coerce")

    df["range"] = df["max"] - df["min"]

    # Choose "most suspicious" sensor for red
    red_sensor = None
    if not shap.empty and "sensor" in shap.columns:
        red_sensor = str(shap.iloc[0]["sensor"])
    else:
        # fallback: largest range
        df2 = df.dropna(subset=["range"]).sort_values("range", ascending=False)
        if not df2.empty:
            red_sensor = str(df2.iloc[0]["sensor"])

    colors = []
    for s in df["sensor"].astype(str).tolist():
        colors.append("red" if (red_sensor is not None and s == red_sensor) else "#f9a825")

    err_plus = (df["max"] - df["mean"]).fillna(0)
    err_minus = (df["mean"] - df["min"]).fillna(0)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["sensor"],
            y=df["mean"],
            marker=dict(color=colors),
            error_y=dict(type="data", symmetric=False, array=err_plus, arrayminus=err_minus),
            hovertemplate="sensor=%{x}<br>mean=%{y}<extra></extra>",
            name="Mean (min/max error bars)",
        )
    )

    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Sensor",
        yaxis_title="Value",
    )
    st.plotly_chart(fig, use_container_width=True)

    if red_sensor:
        st.caption(f"🔴 Highlighted sensor: `{red_sensor}` (SHAP top-1 if available, otherwise largest window range).")


def extract_window_df(a: dict) -> pd.DataFrame:
    """
    Requires consumer to write:
      "window": { "temperature_value": [...], "co2_value": [...], ... }
    """
    w = a.get("window")
    if not isinstance(w, dict):
        return pd.DataFrame()

    cols = {}
    for s in SENSOR_BASE:
        v = w.get(s)
        if isinstance(v, list) and len(v) > 1:
            cols[s] = pd.to_numeric(pd.Series(v), errors="coerce")
    return pd.DataFrame(cols) if cols else pd.DataFrame()


def plot_corr_heatmap(window_df: pd.DataFrame):
    if window_df.empty:
        return
    corr = window_df.corr(numeric_only=True).fillna(0)
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation matrix (window)",
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


def plot_context_scores(detections_df: pd.DataFrame, a: dict, n: int = 250):
    if detections_df.empty or "timestamp" not in detections_df.columns or "if_score" not in detections_df.columns:
        return

    rid = str(a.get("resourceid", ""))
    seq_id = str(a.get("seq_id", "")).strip()

    df = detections_df.copy()
    if "resourceid" in df.columns and rid:
        df = df[df["resourceid"].astype(str) == rid]

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if len(df) > n:
        df = df.iloc[-n:]

    df["if_score"] = pd.to_numeric(df["if_score"], errors="coerce")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["if_score"], mode="lines", name="IF score (recent)"))

    if "seq_id" in df.columns and seq_id:
        df_seq = df[df["seq_id"].astype(str).str.strip() == seq_id]
        if not df_seq.empty:
            fig.add_trace(
                go.Scatter(
                    x=df_seq["timestamp"],
                    y=df_seq["if_score"],
                    mode="markers",
                    name="Current alert",
                    marker=dict(color="red", size=11),
                    hovertemplate="CURRENT<br>time=%{x}<br>score=%{y}<extra></extra>",
                )
            )

    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Time", yaxis_title="IF score")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------
st.sidebar.title("About this dashboard")
st.sidebar.write(
    """
**Pipeline overview**
1. Sensor windows stream from Kafka.
2. Adaptive IF produces score + anomaly flag.
3. Alerts are *locked* for human review.
4. Human labels (true/false) update evaluation.
"""
)
st.sidebar.markdown("---")

detections_df_full = load_detections()
rooms = []
if not detections_df_full.empty and "resourceid" in detections_df_full.columns:
    rooms = sorted(detections_df_full["resourceid"].dropna().astype(str).unique())

selected_room = st.sidebar.selectbox("Filter by room", options=["All rooms"] + rooms if rooms else ["All rooms"], index=0)
n_last = st.sidebar.slider("Recent detections to plot", min_value=50, max_value=3000, value=600, step=50)

auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
refresh_sec = st.sidebar.slider("Refresh interval (seconds)", 1, 10, 1)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: If you feel the UI is too active, disable Auto-refresh while reviewing.")

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
st.title("Real-Time Anomaly Detection – Adaptive Isolation Forest")

labels_df = load_labels()
detections_df = load_detections()

# Apply filters
if selected_room != "All rooms" and not detections_df.empty and "resourceid" in detections_df.columns:
    detections_df = detections_df[detections_df["resourceid"].astype(str) == selected_room]
if len(detections_df) > n_last:
    detections_df = detections_df.iloc[-n_last:]

if selected_room != "All rooms" and not labels_df.empty and "resourceid" in labels_df.columns:
    labels_df = labels_df[labels_df["resourceid"].astype(str) == selected_room]

# -------------------- Status + metrics --------------------
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
        st.info("No labels yet. Label alerts below to compute metrics.")
    else:
        alerts, tp, fp, prec = compute_precision(labels_df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Alerts flagged", alerts)
        c2.metric("True positives", tp)
        c3.metric("False positives", fp)
        c4.metric("Precision", f"{prec:.2f}" if prec == prec else "N/A")

st.divider()

# -------------------- Tabs --------------------
tab_scores, tab_labels, tab_eval = st.tabs(["📈 Scores over time", "📄 Label history", "📐 ROC & PR evaluation"])

with tab_scores:
    st.markdown("#### IF Score vs Threshold")
    st.caption("🟡 = machine anomaly (unconfirmed) • 🔴 = confirmed TRUE • ⚪ = confirmed FALSE")

    if detections_df.empty:
        st.info("No detections available after filters.")
    else:
        det_plot = build_confirmed_points(detections_df, labels_df)
        det_plot = det_plot.dropna(subset=["timestamp"]).sort_values("timestamp")

        det_plot["if_score"] = pd.to_numeric(det_plot.get("if_score"), errors="coerce")
        if "if_threshold" in det_plot.columns:
            det_plot["if_threshold"] = pd.to_numeric(det_plot.get("if_threshold"), errors="coerce")
        if "if_is_anomaly" in det_plot.columns:
            det_plot["if_is_anomaly"] = pd.to_numeric(det_plot.get("if_is_anomaly"), errors="coerce").fillna(0).astype(int)
        else:
            det_plot["if_is_anomaly"] = 0

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=det_plot["timestamp"], y=det_plot["if_score"], mode="lines", name="IF score"))

        if "if_threshold" in det_plot.columns and det_plot["if_threshold"].notna().any():
            fig.add_trace(go.Scatter(x=det_plot["timestamp"], y=det_plot["if_threshold"], mode="lines", name="Threshold", line=dict(dash="dash")))

        machine_anoms = det_plot[det_plot["if_is_anomaly"] == 1]
        machine_only = machine_anoms[(machine_anoms["confirmed_true_anomaly"] == 0) & (machine_anoms["confirmed_false_alarm"] == 0)]
        if not machine_only.empty:
            fig.add_trace(go.Scatter(x=machine_only["timestamp"], y=machine_only["if_score"], mode="markers", name="Machine anomaly (unconfirmed)", marker=dict(color="gold", size=9)))

        confirmed_true = det_plot[det_plot["confirmed_true_anomaly"] == 1]
        if not confirmed_true.empty:
            fig.add_trace(go.Scatter(x=confirmed_true["timestamp"], y=confirmed_true["if_score"], mode="markers", name="Confirmed TRUE", marker=dict(color="red", size=10)))

        confirmed_false = det_plot[det_plot["confirmed_false_alarm"] == 1]
        if not confirmed_false.empty:
            fig.add_trace(go.Scatter(x=confirmed_false["timestamp"], y=confirmed_false["if_score"], mode="markers", name="Confirmed FALSE", marker=dict(color="gray", size=8, symbol="circle-open")))

        fig.update_layout(height=470, xaxis_title="Time", yaxis_title="IF score", margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig, use_container_width=True)

with tab_labels:
    st.markdown("#### Labeled event history")
    if labels_df.empty:
        st.info("No labeled events yet.")
    else:
        if "timestamp" in labels_df.columns:
            labels_df = labels_df.sort_values("timestamp")
        show_cols = [c for c in ["timestamp", "resourceid", "seq_id", "if_is_anomaly", "user_label"] if c in labels_df.columns]
        st.dataframe(labels_df[show_cols].tail(300), use_container_width=True)

with tab_eval:
    st.markdown("#### ROC & Precision–Recall evaluation (Adaptive IF)")
    df_eval = prepare_eval_dataframe(selected_room)
    if df_eval.empty:
        st.info("Not enough labeled data yet. Label more events below.")
    else:
        y_true = df_eval["user_label"].astype(int).values
        s_if = pd.to_numeric(df_eval["if_score"], errors="coerce").values

        res = compute_roc_pr(y_true, s_if)
        if res is None:
            st.info("Need both normal and anomalous labels to compute ROC/PR curves.")
        else:
            best_f1 = best_f1_from_pr(y_true, s_if)
            st.dataframe(pd.DataFrame([{"AUC-ROC": res["roc_auc"], "AUC-PR": res["ap"], "Best F1 (approx)": best_f1}], index=["Adaptive IF"]), use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**ROC curve**")
                st.line_chart(pd.DataFrame({"TPR": res["tpr"]}, index=res["fpr"]).rename_axis("FPR"))
            with c2:
                st.markdown("**Precision–Recall curve**")
                st.line_chart(pd.DataFrame({"Precision": res["precision"]}, index=res["recall"]).rename_axis("Recall"))

st.divider()

# ---------------------------------------------------------------------
# LIVE ANOMALY STREAM (FROZEN FOR REVIEW)
# ---------------------------------------------------------------------
st.subheader("Human-in-the-loop review (Frozen alert)")

now_t = time.time()
if now_t < st.session_state.snooze_until:
    st.info("😴 Snoozed. The current alert is hidden temporarily.")
else:
    latest = read_current_anomaly_json()

    if st.session_state.review_lock is None and latest is not None:
        st.session_state.review_lock = latest

    a = st.session_state.review_lock

    if a is None:
        st.info("Monitoring... waiting for next anomaly from Kafka.")
    else:
        st.markdown(
            """
            <div class="lock-card">
              <div class="lock-title">🔒 LOCKED FOR HUMAN REVIEW</div>
              <div class="lock-sub">
                This alert will NOT change until you label it.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        rid = a.get("resourceid", "")
        ts = a.get("timestamp", "")
        seq_id = str(a.get("seq_id", "")).strip()

        chips = [f'<span class="chip">IF: {"ANOMALY" if a.get("if_is_anomaly") else "normal"}</span>']
        st.markdown(" ".join(chips), unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns([1.2, 1.2, 1, 1])
        k1.metric("Room", str(rid))
        k2.metric("Timestamp", str(ts))
        k3.metric("IF score", a.get("if_score", ""))
        k4.metric("Threshold", a.get("if_threshold", ""))

        left, right = st.columns([1.15, 0.85])

        with left:
            st.markdown("### Context (recent IF score history)")
            plot_context_scores(detections_df_full, a, n=250)

        with right:
            st.markdown("### Explainability (SHAP)")
            df_shap = extract_shap_importance(a)
            if df_shap.empty:
                st.info("No SHAP importance available for this alert yet.")
            else:
                plot_shap_bar(df_shap, top_k=6)
                st.caption("Red bar = strongest driver among sensors (SHAP top-1).")

        st.markdown("### Sensor snapshot (window summary)")
        snap = extract_sensor_snapshot(a)
        df_shap = extract_shap_importance(a)

        if snap.empty:
            st.info("No sensor snapshot in current_anomaly.json. Add *_mean/min/max keys from the consumer.")
        else:
            snap["range"] = pd.to_numeric(snap["max"], errors="coerce") - pd.to_numeric(snap["min"], errors="coerce")
            st.dataframe(snap.sort_values("range", ascending=False), use_container_width=True, hide_index=True)

            st.markdown("#### Sensor bar plot (mean with min/max error bars)")
            plot_sensor_snapshot_bar(snap, df_shap)

        st.markdown("### Feature relationships (correlation)")
        window_df = extract_window_df(a)
        if window_df.empty:
            st.info("Correlation requires full window arrays. Add `window:{sensor:[...]} ` into current_anomaly.json from the consumer.")
        else:
            plot_corr_heatmap(window_df)

        st.markdown("---")
        st.markdown("## Decision")
        ensure_labels_file()

        colA, colB, colC = st.columns([1, 1, 1.2])

        if colA.button("✅ Confirm TRUE anomaly", type="primary"):
            with label_log_path.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([seq_id, rid, ts, int(bool(a.get("if_is_anomaly"))), 1])

            try:
                anomaly_path.unlink(missing_ok=True)
            except Exception:
                pass

            st.session_state.review_lock = None
            st.success("Saved as TRUE anomaly. Ready for next alert.")

        if colB.button("🚫 Mark as FALSE alarm"):
            with label_log_path.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([seq_id, rid, ts, int(bool(a.get("if_is_anomaly"))), 0])

            try:
                anomaly_path.unlink(missing_ok=True)
            except Exception:
                pass

            st.session_state.review_lock = None
            st.success("Saved as FALSE alarm. Ready for next alert.")

        if colC.button("😴 Snooze this alert (10s)"):
            st.session_state.snooze_until = time.time() + 10.0
            st.info("Snoozed for 10 seconds (the alert is still locked).")

# ---------------------------------------------------------------------
# AUTO REFRESH
# ---------------------------------------------------------------------
if auto_refresh:
    time.sleep(refresh_sec)
    st.rerun()
