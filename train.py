import os
import mlflow

# 1. Force creation of the directory
os.makedirs("mlruns", exist_ok=True)

# 2. Use a simple relative path (Works on both Windows and Linux)
mlflow.set_tracking_uri("file:./mlruns")

def run_train():
    with mlflow.start_run() as run:
        # Log your fake accuracy for the assignment
        mlflow.log_metric("accuracy", 0.95)
        
        # Save the Run ID
        run_id = run.info.run_id
        with open("model_info.txt", "w") as f:
            f.write(run_id)
        
        print(f"✅ Success! Run ID: {run_id}")

if __name__ == "__main__":
    run_train()