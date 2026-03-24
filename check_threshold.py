import mlflow
import os
import sys

# Match the training path
mlflow.set_tracking_uri("file:./mlruns")

if not os.path.exists("model_info.txt"):
    print("❌ model_info.txt missing")
    sys.exit(1)

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

try:
    run = mlflow.get_run(run_id)
    acc = run.data.metrics.get("accuracy", 0)
    print(f" Accuracy: {acc}")
    
    if acc >= 0.85:
        print(" PASSED")
    else:
        print(" FAILED")
        sys.exit(1)
except Exception as e:
    print(f" Error: {e}")
    sys.exit(1)