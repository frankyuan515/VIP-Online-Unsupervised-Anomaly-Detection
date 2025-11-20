import json
import time
import random
from datetime import datetime
from kafka import KafkaProducer

# Connect to your existing Kafka on localhost:9092
producer = KafkaProducer(
    bootstrap_servers=["localhost:9092"],
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

SENSOR_COLS = [
    "temperature_value",
    "humidity_value",
    "co2_value",
    "light_value",
    "radon_value",
    "airquality_value",
]

def generate_fake_sensor_data():
    """Generate one synthetic reading for all sensors."""
    return {
        "temperature_value": round(random.uniform(19, 24), 2),
        "humidity_value": round(random.uniform(30, 60), 2),
        "co2_value": round(random.uniform(400, 900), 2),
        "light_value": round(random.uniform(100, 800), 2),
        "radon_value": round(random.uniform(0, 40), 2),
        "airquality_value": round(random.uniform(0, 10), 2),
    }

seq_id = 0
resourceid = "12834a49-ce1d-4f43-949a-996dbb088f74"

print("📡 Streaming synthetic data to Kafka topic 'nimway-sensors'...")
print("Using resourceid:", resourceid)


while True:
    sensors = generate_fake_sensor_data()

    message = {
        "seq_id": seq_id,
        "resourceid": resourceid, 
        "start_time": datetime.now().isoformat(),
        "sensors": sensors,
    }

    producer.send("nimway-sensors", message)
    producer.flush()

    print("Sent:", message)

    seq_id += 1
    time.sleep(1)  # send one message per second
