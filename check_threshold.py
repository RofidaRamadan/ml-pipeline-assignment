# 

import mlflow
import os
import sys

# Look for the folder exactly where it was downloaded
tracking_uri = "file://" + os.path.abspath("mlruns")
mlflow.set_tracking_uri(tracking_uri)

print(f"Looking for data in: {tracking_uri}")

if not os.path.exists("model_info.txt"):
    print("❌ ERROR: model_info.txt is missing!")
    sys.exit(1)

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

try:
    run = mlflow.get_run(run_id)
    acc = run.data.metrics.get("accuracy")
    print(f" Found Accuracy: {acc}")

    if acc >= 0.85:
        print(" THRESHOLD PASSED")
    else:
        print(f" THRESHOLD FAILED: {acc} < 0.85")
        sys.exit(1)
except Exception as e:
    print(f" Error reading MLflow: {e}")
    sys.exit(1)