import os
os.makedirs("mlruns")
import mlflow

def main():
    # 1. Create the mlruns folder if it doesn't exist
    if not os.path.exists("mlruns"):
        os.makedirs("mlruns")

    # 2. Set the tracking URI to the local folder
    tracking_uri = "file://" + os.path.abspath("mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    
    # 3. Start the run
    with mlflow.start_run() as run:
        mlflow.log_metric("accuracy", 0.95)
        
        # 4. Save the Run ID
        run_id = run.info.run_id
        with open("model_info.txt", "w") as f:
            f.write(run_id)
        
        print(f"✅ Success! Accuracy 0.95 logged. ID: {run_id}")

if __name__ == "__main__":
    main()