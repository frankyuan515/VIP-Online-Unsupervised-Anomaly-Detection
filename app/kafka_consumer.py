'''
from kafka import KafkaConsumer
import json
import csv
from pathlib import Path

import numpy as np
import shap

from services.online_trainer import OnlineTrainer
from services.if_trainer import IFOnlineTrainer
from services.preprocessing import StreamPreprocessor, SENSOR_COLS, WINDOW_SIZE

# ---------------- CONFIG ----------------

TOPIC = "nimway-sensors"
BOOTSTRAP_SERVERS = ["localhost:9092"]

# If > 0, every Nth message will be sent to the UI as an "anomaly"
# even if VAE/IF do not flag it.
DEBUG_FORCE_ALERT_EVERY = 10  # set to 0 to disable forcing

# ----------------------------------------

trainer_vae = OnlineTrainer()
trainer_if = IFOnlineTrainer()
preproc = StreamPreprocessor()

# SHAP: we will build an explainer lazily
shap_explainer = None

# Set up CSV logging
log_path = Path("data/detections_log.csv")
log_path.parent.mkdir(parents=True, exist_ok=True)

if not log_path.exists():
    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "seq_id",
                "resourceid",
                "timestamp",
                "vae_error",
                "vae_is_anomaly",
                "vae_threshold",
                "if_score",
                "if_is_anomaly",
                "if_threshold",
            ]
        )

# Kafka consumer
consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=BOOTSTRAP_SERVERS,
    value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    auto_offset_reset="latest",      # start from new messages
    enable_auto_commit=True,
    group_id="vae-if-consumer-v2",   # new group to avoid old offsets
)

print(f"🔄 Kafka consumer started, waiting for messages on '{TOPIC}'...")

msg_count = 0

for msg in consumer:
    msg_count += 1
    data = msg.value

    seq_id = data["seq_id"]
    rid = data["resourceid"]
    ts = data["start_time"]
    sensors_dict = data["sensors"]

    # Build windows for both models
    try:
        seq_torch = preproc.to_torch_window(sensors_dict, rid)  # (1, 60) torch
        seq_np = preproc.to_numpy_window(sensors_dict, rid)      # (1, 60) numpy
    except Exception as e:
        print(f"[WARN] Preprocessing failed for resourceid={rid}: {e}")
        continue

    # VAE
    vae_error, vae_flag, vae_thresh = trainer_vae.update(seq_torch, rid)

    # Guard against NaNs from VAE
    if not np.isfinite(vae_error):
        print(f"[WARN] VAE returned non-finite error (rid={rid}); skipping VAE flag.")
        vae_error = np.nan
        vae_flag = False
        # leave vae_thresh as-is or set to None

    # Adaptive Isolation Forest
    if_score, if_flag, if_thresh = trainer_if.update(seq_np, rid)

    # Clip scores for safety before logging
    if np.isfinite(vae_error):
        vae_error_safe = float(np.clip(vae_error, 0.0, 1e9))
    else:
        vae_error_safe = float("nan")

    if_score_safe = float(np.clip(if_score, -1e9, 1e9))

    # Log row
    with log_path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                seq_id,
                rid,
                ts,
                vae_error_safe,
                int(bool(vae_flag)),
                vae_thresh if vae_thresh is not None else "",
                if_score_safe,
                int(bool(if_flag)),
                if_thresh if if_thresh is not None else "",
            ]
        )

    # Decide if we trigger the UI
    force_alert = (
        DEBUG_FORCE_ALERT_EVERY > 0
        and (msg_count % DEBUG_FORCE_ALERT_EVERY == 0)
    )
    trigger_alert = bool(vae_flag or if_flag or force_alert)

    if trigger_alert:
        # ---- SHAP explainability for Adaptive IF (optional) ----
        if_shap_importance = None
        try:
            base_model = trainer_if.model.get_shap_model()
            explainer = shap.TreeExplainer(base_model)

            # seq_np is (1, WINDOW_SIZE * len(SENSOR_COLS))
            shap_vals = explainer.shap_values(seq_np)  # shape: (1, n_features)
            shap_vals = np.array(shap_vals)[0]         # (n_features,)

            num_sensors = len(SENSOR_COLS)
            shap_2d = shap_vals.reshape(WINDOW_SIZE, num_sensors)

            mean_abs = np.mean(np.abs(shap_2d), axis=0)  # (num_sensors,)
            if_shap_importance = {
                col: float(val) for col, val in zip(SENSOR_COLS, mean_abs)
            }
        except Exception as e:
            print(f"[WARN] SHAP computation failed: {e}")
            if_shap_importance = None

        anomaly_path = Path("data/current_anomaly.json")
        anomaly = {
            "seq_id": seq_id,
            "resourceid": rid,
            "timestamp": ts,
            "vae_error": float(round(vae_error_safe, 3)) if np.isfinite(vae_error_safe) else None,
            "vae_threshold": float(round(vae_thresh, 3)) if vae_thresh is not None else None,
            "if_score": float(round(if_score_safe, 3)),
            "if_threshold": float(round(if_thresh, 3)) if if_thresh is not None else None,
            "vae_is_anomaly": bool(vae_flag),
            "if_is_anomaly": bool(if_flag),
            "if_shap_importance": if_shap_importance,
        }
        anomaly_path.write_text(json.dumps(anomaly))
        print(f"🚨 ALERT → {anomaly}")
    else:
        print(f"✅ Normal: VAE={vae_error_safe}, IF={if_score_safe}")
'''
#only aif
# app/kafka_consumer.py

from kafka import KafkaConsumer
import json
import csv
from pathlib import Path

import numpy as np
import shap

from services.if_trainer import IFOnlineTrainer
from services.preprocessing import StreamPreprocessor, SENSOR_COLS, WINDOW_SIZE

# ---------------- CONFIG ----------------

TOPIC = "nimway-sensors"
BOOTSTRAP_SERVERS = ["localhost:9092"]

# If > 0, every Nth message will be sent to the UI for labeling,
# even if the model does not flag it as an anomaly (useful for debugging & getting normal labels).
DEBUG_FORCE_ALERT_EVERY = 0  # set to e.g. 10 if you want forced events

# ----------------------------------------

trainer_if = IFOnlineTrainer()
preproc = StreamPreprocessor()

# Set up CSV logging: AIF-only
log_path = Path("data/detections_log.csv")
log_path.parent.mkdir(parents=True, exist_ok=True)

if not log_path.exists():
    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "seq_id",
                "resourceid",
                "timestamp",
                "if_score",
                "if_is_anomaly",
                "if_threshold",
            ]
        )

# Kafka consumer
consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=BOOTSTRAP_SERVERS,
    value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    auto_offset_reset="latest",
    enable_auto_commit=True,
    group_id="aif-consumer-v1",
)

print(f"🔄 Kafka consumer started, waiting for messages on '{TOPIC}'...")

msg_count = 0

for msg in consumer:
    msg_count += 1
    data = msg.value

    seq_id = data["seq_id"]
    rid = data["resourceid"]
    ts = data["start_time"]
    sensors_dict = data["sensors"]

    # Build window for AIF
    try:
        seq_np = preproc.to_numpy_window(sensors_dict, rid)  # shape (1, WINDOW_SIZE * len(SENSOR_COLS))
    except Exception as e:
        print(f"[WARN] Preprocessing failed for resourceid={rid}: {e}")
        continue

    # Adaptive Isolation Forest update
    if_score, if_flag, if_thresh = trainer_if.update(seq_np, rid)

    # Clip scores for safety before logging
    if_score_safe = float(np.clip(if_score, -1e9, 1e9))
    if_thresh_safe = float(if_thresh) if if_thresh is not None else ""

    # Log row
    with log_path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                seq_id,
                rid,
                ts,
                if_score_safe,
                int(bool(if_flag)),
                if_thresh_safe,
            ]
        )

    # Decide if we trigger the UI (AIF only + optional forced alerts)
    force_alert = (
        DEBUG_FORCE_ALERT_EVERY > 5
        and (msg_count % DEBUG_FORCE_ALERT_EVERY == 0)
    )
    trigger_alert = bool(if_flag or force_alert)

    if trigger_alert:
        # ---- SHAP explainability for AIF (optional) ----
        if_shap_importance = None
        try:
            base_model = trainer_if.model.get_shap_model()
            explainer = shap.TreeExplainer(base_model)

            shap_vals = explainer.shap_values(seq_np)  # shape (1, n_features)
            shap_vals = np.array(shap_vals)[0]         # (n_features,)

            num_sensors = len(SENSOR_COLS)
            shap_2d = shap_vals.reshape(WINDOW_SIZE, num_sensors)

            mean_abs = np.mean(np.abs(shap_2d), axis=0)  # (num_sensors,)
            if_shap_importance = {
                col: float(val) for col, val in zip(SENSOR_COLS, mean_abs)
            }
        except Exception as e:
            print(f"[WARN] SHAP computation failed: {e}")
            if_shap_importance = None

        anomaly_path = Path("data/current_anomaly.json")
        anomaly = {
            "seq_id": seq_id,
            "resourceid": rid,
            "timestamp": ts,
            "if_score": float(round(if_score_safe, 3)),
            "if_threshold": float(round(if_thresh, 3))
            if if_thresh is not None
            else None,
            "if_is_anomaly": bool(if_flag),
            "if_shap_importance": if_shap_importance,
        }
        anomaly_path.write_text(json.dumps(anomaly))
        print(f"🚨 ALERT → {anomaly}")
    else:
        
        print(f"✅ Normal: IF={if_score_safe}, thresh={if_thresh_safe}")

