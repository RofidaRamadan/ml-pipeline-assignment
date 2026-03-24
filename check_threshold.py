import mlflow
import os
import sys
import platform

path = os.path.abspath("mlruns")
if platform.system() == "Windows":
    tracking_uri = f"file:///{path.replace(os.sep, '/')}"
else:
    tracking_uri = f"file://{path}"

mlflow.set_tracking_uri(tracking_uri)

if not os.path.exists("model_info.txt"):
    print("❌ Error: model_info.txt not found")
    sys.exit(1)

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

try:
    run = mlflow.get_run(run_id)
    acc = run.data.metrics.get("accuracy")
    print(f"📊 Accuracy: {acc}")
    if acc >= 0.85:
        print("🚀 PASSED")
    else:
        sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)