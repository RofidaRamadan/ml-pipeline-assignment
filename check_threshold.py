# # # import mlflow
# # # import os
# # # import sys

# # # def check_model_performance():
# # #     # 1. Set the Tracking URI
# # #     default_local_path = "file:///C:/Users/RofaR/OneDrive/Desktop/my-mlops-project/mlruns"
# # #     tracking_uri = os.getenv("MLFLOW_TRACKING_URI", default_local_path)
# # #     mlflow.set_tracking_uri(tracking_uri)

# # #     # 2. Read the Run ID from the file created by train.py
# # #     if not os.path.exists("model_info.txt"):
# # #         print("ERROR: model_info.txt not found. Did you run train.py first?")
# # #         sys.exit(1)
        
# # #     with open("model_info.txt", "r") as f:
# # #         run_id = f.read().strip()

# # #     # 3. Fetch the results from MLflow
# # #     try:
# # #         run = mlflow.get_run(run_id)
# # #         # We fetch the latest 'accuracy' metric
# # #         accuracy = run.data.metrics.get("accuracy", 0)
# # #         print(f"--- Threshold Check ---")
# # #         print(f"Run ID: {run_id}")
# # #         print(f"Model Accuracy: {accuracy:.4f}")
# # #     except Exception as e:
# # #         print(f"ERROR: Could not find Run ID {run_id} in MLflow. {e}")
# # #         sys.exit(1)

# # #     # 4. The Logic Gate (Requirement: Fail if < 0.85)
# # #     threshold = 0.85
# # #     if accuracy < threshold:
# # #         print(f"RESULT: FAILED. Accuracy {accuracy:.4f} is below {threshold}.")
# # #         # Exiting with 1 tells GitHub Actions to stop the pipeline and show a Red X
# # #         sys.exit(1)
# # #     else:
# # #         print(f"RESULT: SUCCESS! Accuracy {accuracy:.4f} meets the threshold.")
# # #         # Exiting with 0 tells GitHub Actions everything is perfect
# # #         sys.exit(0)

# # # if __name__ == "__main__":
# # #     check_model_performance()

# # import mlflow
# # import os

# # if __name__ == "__main__":
# #     # 1. Point to the same local folder
# #     mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))

# #     # 2. Read the Run ID from the file
# #     with open("model_info.txt", "r") as f:
# #         run_id = f.read().strip()

# #     # 3. Fetch the metric
# #     run = mlflow.get_run(run_id)
# #     accuracy = run.data.metrics.get("accuracy")

# #     print(f"Checking Run {run_id} - Accuracy: {accuracy}")

# #     if accuracy >= 0.85:
# #         print("SUCCESS: Model passed the threshold!")
# #     else:
# #         print("FAILED: Accuracy too low.")
# #         exit(1)
        
# # import mlflow
# # import os

# # # 1. Look at the local folder we just downloaded
# # mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))

# # # 2. Get the Run ID from the text file
# # with open("model_info.txt", "r") as f:
# #     run_id = f.read().strip()

# # # 3. Pull the accuracy
# # run = mlflow.get_run(run_id)
# # accuracy = run.data.metrics.get("accuracy")

# # print(f"Checking Run ID: {run_id}")
# # print(f"Model Accuracy: {accuracy}")

# # # 4. The Logic Check
# # if accuracy is not None and accuracy >= 0.85:
# #     print("SUCCESS: Accuracy meets threshold. Deploying...")
# # else:
# #     print(f" FAILED: Accuracy {accuracy} is too low or missing.")
# #     exit(1)

# import mlflow
# import os
# import sys

# # 1. Point to the folder downloaded from the artifact
# # This ensures we see the data from the 'validate' job
# tracking_path = os.path.join(os.getcwd(), "mlruns")
# mlflow.set_tracking_uri("file://" + tracking_path)

# try:
#     # 2. Read the Run ID saved during training
#     if not os.path.exists("model_info.txt"):
#         print("ERROR: model_info.txt not found!")
#         sys.exit(1)
        
#     with open("model_info.txt", "r") as f:
#         run_id = f.read().strip()

#     # 3. Get the model results from MLflow
#     run = mlflow.get_run(run_id)
#     accuracy = run.data.metrics.get("accuracy")

#     print("--- Pipeline Results ---")
#     print(f"Checking Run ID: {run_id}")
#     print(f"Model Accuracy:  {accuracy}")
#     print("------------------------")

#     # 4. Threshold Logic (Assignment requires 0.85)
#     threshold = 0.85
#     if accuracy is not None and accuracy >= threshold:
#         print(f" SUCCESS: Accuracy {accuracy} is above {threshold}. Deploying model...")
#     else:
#         print(f" FAILED: Accuracy {accuracy} is below {threshold}. Blocking deployment.")
#         sys.exit(1)

# except Exception as e:
#     print(f"An error occurred: {e}")
#     sys.exit(1)

import mlflow
import os
import sys

# Look for mlruns in the current folder
tracking_path = os.path.abspath("mlruns")
mlflow.set_tracking_uri("file://" + tracking_path)

try:
    if not os.path.exists("model_info.txt"):
        print(" ERROR: model_info.txt was not found in the deploy job!")
        sys.exit(1)

    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()

    run = mlflow.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy")

    print(f"--- VALIDATING RUN: {run_id} ---")
    print(f"Accuracy Found: {accuracy}")

    if accuracy is not None and accuracy >= 0.85:
        print(" SUCCESS: Accuracy passed threshold!")
    else:
        print(f" FAILED: Accuracy {accuracy} is too low.")
        sys.exit(1)

except Exception as e:
    print(f" Error: {e}")
    sys.exit(1)