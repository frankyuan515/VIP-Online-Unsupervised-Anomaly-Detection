
import pandas as pd
import json
import time
from kafka import KafkaProducer
from pathlib import Path

CSV_PATH = Path("data/vae_final_streaming.csv")
TOPIC = "nimway-sensors"

print(f"Loading {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)
print(f"   {len(df):,} sequences")

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    retries=5,
    request_timeout_ms=30000,
)

SENSOR_COLS = [
    "temperature_value",
    "humidity_value",
    "co2_value",
    "light_value",
    "radon_value",
    "airquality_value",
]
WINDOW_SIZE = 10

print("Streaming...")
for _, row in df.iterrows():
    sensors = {}
    for t in range(WINDOW_SIZE):
        for col in SENSOR_COLS:
            sensors[f"{col}_t{t}"] = float(row[f"{col}_t{t}"])

    payload = {
        "seq_id": str(row["seq_id"]),
        "resourceid": row["resourceid"],
        "start_time": str(row["start_time"]),
        "sensors": sensors,
    }
    producer.send(TOPIC, payload)
    print(f"Sent: {row['seq_id']}")
    time.sleep(0.1)

print("Done.")
