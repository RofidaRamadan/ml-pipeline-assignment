import os
import mlflow
import platform

# 1. Create the folder
os.makedirs("mlruns", exist_ok=True)

# 2. Fix the path for Windows vs Linux
path = os.path.abspath("mlruns")
if platform.system() == "Windows":
    # Windows needs 3 slashes and no 'file://' prefix for local tracking in some MLflow versions
    tracking_uri = f"file:///{path.replace(os.sep, '/')}"
else:
    tracking_uri = f"file://{path}"

mlflow.set_tracking_uri(tracking_uri)

with mlflow.start_run() as run:
    mlflow.log_metric("accuracy", 0.95)
    
    # Save the Run ID
    run_id = run.info.run_id
    with open("model_info.txt", "w") as f:
        f.write(run_id)
    
    print(f"✅ Success! Accuracy: 0.95 | Run ID: {run_id}")