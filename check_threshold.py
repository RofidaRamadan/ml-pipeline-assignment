import mlflow
import os
import sys

def check_model_performance():
    # 1. Set the Tracking URI
    default_local_path = "file:///C:/Users/RofaR/OneDrive/Desktop/my-mlops-project/mlruns"
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", default_local_path)
    mlflow.set_tracking_uri(tracking_uri)

    # 2. Read the Run ID from the file created by train.py
    if not os.path.exists("model_info.txt"):
        print("ERROR: model_info.txt not found. Did you run train.py first?")
        sys.exit(1)
        
    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()

    # 3. Fetch the results from MLflow
    try:
        run = mlflow.get_run(run_id)
        # We fetch the latest 'accuracy' metric
        accuracy = run.data.metrics.get("accuracy", 0)
        print(f"--- Threshold Check ---")
        print(f"Run ID: {run_id}")
        print(f"Model Accuracy: {accuracy:.4f}")
    except Exception as e:
        print(f"ERROR: Could not find Run ID {run_id} in MLflow. {e}")
        sys.exit(1)

    # 4. The Logic Gate (Requirement: Fail if < 0.85)
    threshold = 0.85
    if accuracy < threshold:
        print(f"RESULT: FAILED. Accuracy {accuracy:.4f} is below {threshold}.")
        # Exiting with 1 tells GitHub Actions to stop the pipeline and show a Red X
        sys.exit(1)
    else:
        print(f"RESULT: SUCCESS! Accuracy {accuracy:.4f} meets the threshold.")
        # Exiting with 0 tells GitHub Actions everything is perfect
        sys.exit(0)

if __name__ == "__main__":
    check_model_performance()