# inspect_scalers.py
import pickle

path = "models/scaler_per_room.pkl"

print("🔎 Loading:", path)

with open(path, "rb") as f:
    scalers = pickle.load(f)

print("\n📌 Rooms found in scaler_per_room.pkl:\n")
for rid in scalers.keys():
    print(" -", rid)

print("\nTotal rooms:", len(scalers))
