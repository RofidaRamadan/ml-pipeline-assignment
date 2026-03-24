import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import mlflow
import mlflow.tensorflow
import os

# --- MODEL BUILDING FUNCTIONS ---
def load_and_preprocess_data():
    (X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.astype('float32')
    X_train = (X_train - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=-1)
    return X_train

def build_generator():
    model = tf.keras.Sequential([
        # Start: 100 noise vector -> 7x7x128
        layers.Input(shape=(100,)),
        layers.Dense(7*7*128, use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((7, 7, 128)),
        
        # Layer 1: Upsample to 14x14
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        
        # Layer 2: Upsample to 28x28 
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model

def build_discriminator():
    model = tf.keras.Sequential([
        # Expects 28x28x1
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# --- THE TRAINING STEP ---
@tf.function
def train_step(images, generator, discriminator, gen_optimizer, disc_optimizer, batch_size, noise_dim):
    noise = tf.random.normal([batch_size, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        disc_loss = real_loss + fake_loss

    gen_optimizer.apply_gradients(zip(gen_tape.gradient(gen_loss, generator.trainable_variables), generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_tape.gradient(disc_loss, discriminator.trainable_variables), discriminator.trainable_variables))
    
    real_acc = tf.reduce_mean(tf.cast(real_output > 0, tf.float32))
    fake_acc = tf.reduce_mean(tf.cast(fake_output < 0, tf.float32))
    return (real_acc + fake_acc) / 2

if __name__ == "__main__":
    # Setup MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:///C:/Users/RofaR/OneDrive/Desktop/my-mlops-project/mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    
    experiment_name = "Assignment5_Rofida"
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    # Params
    BATCH_SIZE = 128
    EPOCHS = 2 
    noise_dim = 100
    
    # Prepare Data
    dataset = tf.data.Dataset.from_tensor_slices(load_and_preprocess_data()).shuffle(60000).batch(BATCH_SIZE)
    generator, discriminator = build_generator(), build_discriminator()
    gen_opt = tf.keras.optimizers.Adam(1e-4)
    disc_opt = tf.keras.optimizers.Adam(1e-4)

    with mlflow.start_run() as run:
        for epoch in range(EPOCHS):
            accuracies = []
            for batch in dataset:
                acc = train_step(batch, generator, discriminator, gen_opt, disc_opt, BATCH_SIZE, noise_dim)
                accuracies.append(acc.numpy())
            
            avg_acc = np.mean(accuracies)
            mlflow.log_metric("accuracy", float(avg_acc), step=epoch)
            print(f"Epoch {epoch+1} - Accuracy: {avg_acc:.4f}")

        # Save model and ID
        mlflow.tensorflow.log_model(generator, "model")
        run_id = run.info.run_id
        
        with open("model_info.txt", "w") as f:
            f.write(run_id)
        
        print(f"DONE. RUN_ID: {run_id}")