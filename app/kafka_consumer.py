from kafka import KafkaConsumer
import json
import csv
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import shap

from services.if_trainer import IFOnlineTrainer
from services.preprocessing import StreamPreprocessor, SENSOR_COLS, WINDOW_SIZE

# ---------------- CONFIG ----------------
TOPIC = "nimway-sensors"
BOOTSTRAP_SERVERS = ["localhost:9092"]

# If > 0, every Nth message will be sent to the UI for labeling
DEBUG_FORCE_ALERT_EVERY = 0  # e.g. 10 for debugging

# Paths
DATA_DIR = Path("data")
DETECTIONS_CSV = DATA_DIR / "detections_log.csv"
CURRENT_ANOMALY_JSON = DATA_DIR / "current_anomaly.json"

# ---------------- INIT ----------------
DATA_DIR.mkdir(parents=True, exist_ok=True)

trainer_if = IFOnlineTrainer()
preproc = StreamPreprocessor()

# Rolling buffers per room: keep last WINDOW_SIZE values per sensor
buffers = defaultdict(lambda: {col: deque(maxlen=WINDOW_SIZE) for col in SENSOR_COLS})

# -------- CSV header: ALWAYS consistent --------
# We log these columns for every row:
# base cols + per-sensor mean/min/max
BASE_COLS = [
    "seq_id",
    "resourceid",
    "timestamp",
    "if_score",
    "if_is_anomaly",
    "if_threshold",
]

SENSOR_STAT_COLS = []
for col in SENSOR_COLS:
    SENSOR_STAT_COLS += [f"{col}_mean", f"{col}_min", f"{col}_max"]

ALL_COLS = BASE_COLS + SENSOR_STAT_COLS

# If file doesn't exist, create it with header
if not DETECTIONS_CSV.exists():
    with DETECTIONS_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(ALL_COLS)

# Kafka consumer
consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=BOOTSTRAP_SERVERS,
    value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    auto_offset_reset="latest",
    enable_auto_commit=True,
    group_id="aif-consumer-v2",
)

print(f"🔄 Kafka consumer started. Topic='{TOPIC}'")

msg_count = 0

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def window_stats(arr: np.ndarray):
    """Return (mean, min, max) ignoring NaNs. If all NaN -> (nan,nan,nan)."""
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return (np.nan, np.nan, np.nan)
    return (float(np.nanmean(arr)), float(np.nanmin(arr)), float(np.nanmax(arr)))

for msg in consumer:
    msg_count += 1
    data = msg.value

    # Expected structure:
    # {
    #   "seq_id": ...,
    #   "resourceid": ...,
    #   "start_time": ...,
    #   "sensors": { "temperature_value": <scalar or list>, ... }
    # }
    try:
        seq_id = str(data.get("seq_id", "")).strip()
        rid = str(data.get("resourceid", "")).strip()
        ts = data.get("start_time") or data.get("timestamp") or ""
        sensors_dict = data.get("sensors", {}) or {}
    except Exception as e:
        print(f"[WARN] Bad message format: {e}")
        continue

    # ------------------------------------------------------------
    # 1) Update rolling buffers (per room) with CURRENT readings
    # ------------------------------------------------------------
    # If sensors_dict has lists -> use last element as "current reading"
    for col in SENSOR_COLS:
        raw = sensors_dict.get(col, np.nan)

        if isinstance(raw, (list, tuple, np.ndarray)):
            if len(raw) == 0:
                v = np.nan
            else:
                v = safe_float(raw[-1])
        else:
            v = safe_float(raw)

        buffers[rid][col].append(v)

    # ------------------------------------------------------------
    # 2) Build model input window (uses your StreamPreprocessor)
    # ------------------------------------------------------------
    try:
        seq_np = preproc.to_numpy_window(sensors_dict, rid)
        # expected: shape (1, WINDOW_SIZE * len(SENSOR_COLS)) or similar
    except Exception as e:
        print(f"[WARN] Preprocessing failed for resourceid={rid}: {e}")
        continue

    # ------------------------------------------------------------
    # 3) Update + score AIF
    # ------------------------------------------------------------
    if_score, if_flag, if_thresh = trainer_if.update(seq_np, rid)

    if_score_safe = float(np.clip(if_score, -1e9, 1e9))
    if_thresh_safe = float(if_thresh) if if_thresh is not None else np.nan
    if_is_anomaly = int(bool(if_flag))

    # ------------------------------------------------------------
    # 4) Compute sensor snapshot stats from rolling buffer
    # ------------------------------------------------------------
    stat_row = {}
    for col in SENSOR_COLS:
        arr = np.array(buffers[rid][col], dtype=float)
        m, mn, mx = window_stats(arr)
        stat_row[f"{col}_mean"] = m
        stat_row[f"{col}_min"] = mn
        stat_row[f"{col}_max"] = mx

    # Flatten stats into CSV order
    stats_list = [stat_row[c] for c in SENSOR_STAT_COLS]

    # ------------------------------------------------------------
    # 5) Append to detections_log.csv (consistent column count!)
    # ------------------------------------------------------------
    with DETECTIONS_CSV.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [seq_id, rid, ts, if_score_safe, if_is_anomaly, if_thresh_safe] + stats_list
        )

    # ------------------------------------------------------------
    # 6) Trigger UI event (anomaly or forced debug)
    # ------------------------------------------------------------
    force_alert = (DEBUG_FORCE_ALERT_EVERY and (msg_count % DEBUG_FORCE_ALERT_EVERY == 0))
    trigger_alert = bool(if_flag or force_alert)

    if trigger_alert:
        # ---- SHAP explainability (optional) ----
        if_shap_importance = None
        try:
            base_model = trainer_if.model.get_shap_model()
            explainer = shap.TreeExplainer(base_model)

            shap_vals = explainer.shap_values(seq_np)  # (1, n_features)
            shap_vals = np.array(shap_vals)[0]         # (n_features,)

            # Try to reshape to (WINDOW_SIZE, num_sensors)
            num_sensors = len(SENSOR_COLS)
            shap_2d = shap_vals.reshape(WINDOW_SIZE, num_sensors)

            mean_abs = np.mean(np.abs(shap_2d), axis=0)
            if_shap_importance = {col: float(v) for col, v in zip(SENSOR_COLS, mean_abs)}
        except Exception as e:
            print(f"[WARN] SHAP computation failed: {e}")
            if_shap_importance = None

        anomaly = {
            "seq_id": seq_id,
            "resourceid": rid,
            "timestamp": ts,
            "if_score": float(round(if_score_safe, 6)),
            "if_threshold": float(round(if_thresh_safe, 6)) if np.isfinite(if_thresh_safe) else None,
            "if_is_anomaly": bool(if_flag),
            "if_shap_importance": if_shap_importance,
        }

        

        # attach sensor stats into the anomaly JSON
        anomaly.update(stat_row)

        anomaly["window"] = {k: sensors_dict.get(k, None) for k in SENSOR_COLS}
        

        CURRENT_ANOMALY_JSON.write_text(json.dumps(anomaly))
        print(f"🚨 ALERT → {anomaly}")
    else:
        print(f"✅ Normal: room={rid} score={if_score_safe:.6f} thresh={if_thresh_safe}")
