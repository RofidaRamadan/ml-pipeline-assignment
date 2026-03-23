FROM python:3.10-slim

# Accept the Run ID as a build argument
ARG RUN_ID
ENV MODEL_RUN_ID=$RUN_ID

WORKDIR /app

# Simulate downloading the model
RUN echo "Downloading model weights for MLflow Run: ${MODEL_RUN_ID}..."

CMD ["echo", "Model Service is Running"]