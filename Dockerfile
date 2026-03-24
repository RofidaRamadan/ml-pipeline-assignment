# 1. Base image
FROM python:3.10-slim

# 2. Accept the RUN_ID as an argument
ARG RUN_ID
ENV MODEL_RUN_ID=${RUN_ID}

# 3. Create a working directory
WORKDIR /app

# 4. Simulate downloading the model
RUN echo "Downloading model for Run ID: ${MODEL_RUN_ID}..." && \
    echo "Model downloaded successfully." > model_status.txt

CMD ["cat", "model_status.txt"]