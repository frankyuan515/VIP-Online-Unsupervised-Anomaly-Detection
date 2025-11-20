#kafka_consumer.py had been modified for adding Adaptive isolation forest.
from kafka import KafkaConsumer
import json
from pathlib import Path
import csv
from services.online_trainer import OnlineTrainer
from services.if_trainer import IFOnlineTrainer
from services.preprocessing import StreamPreprocessor

trainer_vae = OnlineTrainer()
trainer_if = IFOnlineTrainer()
preproc = StreamPreprocessor()

consumer = KafkaConsumer(
    "nimway-sensors",
    bootstrap_servers=["localhost:9092"],
    value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="vae-if-consumer",
)

log_path = Path("data/detections_log.csv")
if not log_path.exists():
    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seq_id", "resourceid", "timestamp",
            "vae_error", "vae_is_anomaly", "vae_threshold",
            "if_score", "if_is_anomaly", "if_threshold",
        ])

for msg in consumer:
    data = msg.value
    seq_id = data["seq_id"]
    rid = data["resourceid"]
    ts = data["start_time"]
    sensors_dict = data["sensors"]

    # 1) Build windows
    seq_torch = preproc.to_torch_window(sensors_dict, rid)
    seq_np = preproc.to_numpy_window(sensors_dict, rid)

    # 2) VAE
    vae_error, vae_flag, vae_thresh = trainer_vae.update(seq_torch, rid)

    # 3) Isolation Forest
    if_score, if_flag, if_thresh = trainer_if.update(seq_np, rid)

    # 4) Write joint log row
    with log_path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            seq_id, rid, ts,
            vae_error, int(vae_flag), vae_thresh,
            if_score, int(if_flag), if_thresh,
        ])

    # 5) Optionally still trigger UI anomalies (you can pick one model or both)
    if vae_flag or if_flag:
        anomaly = {
            "seq_id": seq_id,
            "resourceid": rid,
            "timestamp": ts,
            "vae_error": round(vae_error, 3),
            "vae_threshold": round(vae_thresh, 3),
            "if_score": round(if_score, 3),
            "if_threshold": round(if_thresh, 3),
            "vae_is_anomaly": bool(vae_flag),
            "if_is_anomaly": bool(if_flag),
        }
        Path("data/current_anomaly.json").write_text(json.dumps(anomaly))
        print(f"ALERT → {anomaly}")
    else:
        print(f"Normal: VAE={vae_error:.3f}, IF={if_score:.3f}")
