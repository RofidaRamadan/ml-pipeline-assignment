import mlflow
import os
import sys

# Point to the downloaded folder
tracking_uri = "file://" + os.path.abspath("mlruns")
mlflow.set_tracking_uri(tracking_uri)

if not os.path.exists("model_info.txt"):
    print("❌ ERROR: model_info.txt missing")
    sys.exit(1)

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

try:
    run = mlflow.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy")
    print(f"Checking Accuracy: {accuracy}")

    if accuracy >= 0.85:
        print("🚀 PASSED")
    else:
        print("❌ FAILED")
        sys.exit(1)
except Exception as e:
    print(f"❌ MLflow Error: {e}")
    sys.exit(1)