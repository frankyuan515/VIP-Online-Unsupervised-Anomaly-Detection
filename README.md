        ┌──────────────────────────┐
        │   Sensor Data / Sim.    │
        │ (rooms, desks, IoT)     │
        └──────────┬──────────────┘
                   │  JSON events
                   ▼
            ┌───────────────┐
            │   Kafka       │
            │ topic: nimway-│
            │   sensors     │
            └──────┬────────┘
                   │
                   ▼
        ┌──────────────────────────┐
        │  Kafka Consumer          │
        │  + StreamPreprocessor    │
        │  - per-room windows      │
        │  - scaling               │
        └──────────┬──────────────┘
                   │ seq_np
                   ▼
        ┌──────────────────────────┐
        │ IFOnlineTrainer          │
        │ + AdaptiveIsolationForest│
        │ - per-room model         │
        │ - sliding score history  │
        │ - threshold (percentile) │
        └──────┬─────────┬────────┘
               │         │
               │         │ anomalies
               │         ▼
               │   ┌───────────────┐
               │   │ current_      │
               │   │ anomaly.json  │
               │   └───────────────┘
               │
               │ all detections
               ▼
        ┌──────────────────────────┐
        │ detections_log.csv       │
        └──────────┬──────────────┘
                   │
                   ▼
        ┌──────────────────────────┐
        │  Streamlit Dashboard     │
        │  - real-time charts      │
        │  - live anomaly card     │
        │  - SHAP explanation      │
        │  - True/False labeling   │
        └──────────┬──────────────┘
                   │ labels
                   ▼
        ┌──────────────────────────┐
        │   labels_log.csv         │
        │   + evaluation (ROC/PR)  │
        └──────────────────────────┘
