import mlflow
import os
import sys

# Look for the mlruns folder we downloaded
tracking_path = os.path.abspath("mlruns")
mlflow.set_tracking_uri("file://" + tracking_path)

if not os.path.exists("model_info.txt"):
    print(" ERROR: model_info.txt not found!")
    sys.exit(1)

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

try:
    run = mlflow.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy")
    print(f" Checking Accuracy: {accuracy}")

    if accuracy is not None and accuracy >= 0.85:
        print(" SUCCESS: Threshold passed!")
    else:
        print(f" FAILED: Accuracy {accuracy} is too low.")
        sys.exit(1)
except Exception as e:
    print(f" Error: {e}")
    sys.exit(1)