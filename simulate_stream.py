import json
import time
import random
from datetime import datetime
from kafka import KafkaProducer
import pickle
from pathlib import Path

# ------------------------------------------------------------
# Load valid ROOM_IDS from scaler_per_room.pkl
# ------------------------------------------------------------
scaler_path = Path("models/scaler_per_room.pkl")

if not scaler_path.exists():
    raise FileNotFoundError(
        f"Could not find {scaler_path}. "
        "Make sure you run this script from the project root "
        "and that the scaler file exists."
    )

with scaler_path.open("rb") as f:
    scalers = pickle.load(f)

ROOM_IDS = list(scalers.keys())

if not ROOM_IDS:
    raise RuntimeError("No rooms found in scaler_per_room.pkl")

print("📌 Rooms loaded from scaler_per_room.pkl:")
for rid in ROOM_IDS:
    print(" -", rid)
print(f"Total rooms: {len(ROOM_IDS)}")

# ------------------------------------------------------------
# Kafka Producer
# ------------------------------------------------------------
producer = KafkaProducer(
    bootstrap_servers=["localhost:9092"],
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

TOPIC = "nimway-sensors"

SENSOR_COLS = [
    "temperature_value",
    "humidity_value",
    "co2_value",
    "light_value",
    "radon_value",
    "airquality_value",
]


def generate_fake_sensor_data(is_anomaly: bool = False):
    """
    Generate synthetic sensor data.
    If is_anomaly=True, inject clearly abnormal values.
    """
    data = {
        "temperature_value": round(random.uniform(19, 24), 2),
        "humidity_value": round(random.uniform(30, 60), 2),
        "co2_value": round(random.uniform(400, 900), 2),
        "light_value": round(random.uniform(100, 800), 2),
        "radon_value": round(random.uniform(0, 40), 2),
        "airquality_value": round(random.uniform(0, 10), 2),
    }

    # Inject anomaly sometimes
    if is_anomaly:
        data["co2_value"] = round(random.uniform(2000, 5000), 2)
        data["temperature_value"] = round(random.uniform(28, 40), 2)
        data["humidity_value"] = round(random.uniform(0, 10), 2)

    return data


print(f"\n📡 Streaming synthetic data to Kafka topic '{TOPIC}'...")
print("Rooms in use:")
for r in ROOM_IDS:
    print(" -", r)
print()

seq_id = 0
i = 0

# ------------------------------------------------------------
# Main loop — cycle through rooms and send messages
# ------------------------------------------------------------
while True:
    seq_id += 1

    # Cycle through all valid rooms (round-robin)
    resourceid = ROOM_IDS[i % len(ROOM_IDS)]
    i += 1

    # ~12% of messages anomalous
    is_anom = random.random() < 0.12

    sensors = generate_fake_sensor_data(is_anomaly=is_anom)

    message = {
        "seq_id": seq_id,
        "resourceid": resourceid,
        "start_time": datetime.now().isoformat(),
        "sensors": sensors,
    }

    producer.send(TOPIC, value=message)
    producer.flush()

    status = "🚨 ANOMALY" if is_anom else "Sent normal"
    print(f"{status} → {resourceid} | seq_id={seq_id}")

    time.sleep(1)  # send one message per second
