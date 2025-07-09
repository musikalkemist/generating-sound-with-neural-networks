from tensorflow.keras.datasets import mnist  # Importing MNIST dataset from Keras
from autoencoder import Autoencoder  # Importing the custom Autoencoder class

# Set training hyperparameters
LEARNING_RATE = 0.0005  # Learning rate for optimizer
BATCH_SIZE = 32         # Number of samples per batch
EPOCHS = 500            # Number of epochs to train for

# Function to load and preprocess the MNIST dataset
def load_mnist():
    """
    Load and preprocess the MNIST dataset.
    The pixel values are normalized by dividing by 255.
    The images are reshaped to include a channel dimension (28, 28, 1).
    
    Returns:
        x_train (numpy.ndarray): Training images.
        y_train (numpy.ndarray): Training labels.
        x_test (numpy.ndarray): Test images.
        y_test (numpy.ndarray): Test labels.
    """
    # Load the MNIST dataset from Keras
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize the images to the range [0, 1] by dividing by 255
    x_train = x_train.astype("float32") / 255
    x_train = x_train.reshape(x_train.shape + (1,))  # Reshape to (28, 28, 1) to add a channel dimension
    x_test = x_test.astype("float32") / 255
    x_test = x_test.reshape(x_test.shape + (1,))  # Reshape to (28, 28, 1)

    return x_train, y_train, x_test, y_test

# Function to train the autoencoder model
def train(x_train, learning_rate, batch_size, epochs):
    """
    Train the autoencoder model using the provided training data.

    Parameters:
        x_train (numpy.ndarray): Training images.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Number of samples per batch.
        epochs (int): Number of epochs to train the model.

    Returns:
        autoencoder (Autoencoder): Trained autoencoder model.
    """
    # Initialize the autoencoder with the specified architecture
    autoencoder = Autoencoder(
        input_shape=(28, 28, 1),  # Input shape is (28, 28, 1) corresponding to MNIST images
        conv_filters=(32, 64, 64, 64),  # Number of convolutional filters in each layer
        conv_kernels=(3, 3, 3, 3),  # Size of the convolutional kernels
        conv_strides=(1, 2, 2, 1),  # Strides for each convolutional layer
        latent_space_dim=2  # Dimension of the latent space (bottleneck layer)
    )
    
    # Print a summary of the autoencoder architecture
    autoencoder.summary()

    # Compile the autoencoder model with the given learning rate
    autoencoder.compile(learning_rate)
    
    # Train the autoencoder using the provided data, batch size, and number of epochs
    autoencoder.train(x_train, batch_size, epochs)
    
    return autoencoder  # Return the trained autoencoder model

def train_plotLoss(x_train, learning_rate, batch_size, epochs):
    """
    Train the autoencoder model using the provided training data.

    Parameters:
        x_train (numpy.ndarray): Training images.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Number of samples per batch.
        epochs (int): Number of epochs to train the model.

    Returns:
        autoencoder (Autoencoder): Trained autoencoder model.
        history (dict): Training history containing loss values over epochs.
    """
    # Initialize the autoencoder with the specified architecture
    autoencoder = Autoencoder(
        input_shape=(28, 28, 1),  # Input shape is (28, 28, 1) corresponding to MNIST images
        conv_filters=(32, 64, 64, 64),  # Number of convolutional filters in each layer
        conv_kernels=(3, 3, 3, 3),  # Size of the convolutional kernels
        conv_strides=(1, 2, 2, 1),  # Strides for each convolutional layer
        latent_space_dim=2  # Dimension of the latent space (bottleneck layer)
    )
    
    # Print a summary of the autoencoder architecture
    autoencoder.summary()

    # Compile the autoencoder model with the given learning rate
    autoencoder.compile(learning_rate)
    
    # Train the autoencoder using the provided data, batch size, and number of epochs
    history = autoencoder.train(x_train, batch_size, epochs)
    
    return autoencoder, history  # Return the trained autoencoder model and the training history

import matplotlib.pyplot as plt

def plot_loss_curve(history):
    """
    Plot the training loss curve.
    
    Parameters:
        history (dict): Dictionary containing the loss values over epochs.
    """
    plt.plot(history['loss'], label='Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()



# Main execution
if __name__ == "__main__":
    # Load the MNIST dataset (we only use the training images for autoencoder training)
    x_train, _, _, _ = load_mnist()

    # Train the autoencoder on a subset of the training data (first 10,000 samples)
    # autoencoder = train(x_train[:10000], LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder, history = train_plotLoss(x_train[:10000], LEARNING_RATE, BATCH_SIZE, EPOCHS)
    # Plot the loss curve
    plot_loss_curve(history.history)
    
    # Save the trained autoencoder model to a file
    autoencoder.save("model")
