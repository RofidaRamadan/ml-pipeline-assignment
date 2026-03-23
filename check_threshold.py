import mlflow
import os
import sys

def check():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    
    # 1. Read the Run ID from the artifact
    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()

    # 2. Get the specific run from MLflow
    run = mlflow.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy", 0)
    
    print(f"Checking Run {run_id}: Accuracy = {accuracy:.4f}")
    
    # 3. Fail if below 0.85
    if accuracy < 0.85:
        print("FAILED: Accuracy is below 0.85 threshold.")
        sys.exit(1)
    
    print("SUCCESS: Accuracy threshold met!")

if __name__ == "__main__":
    check()