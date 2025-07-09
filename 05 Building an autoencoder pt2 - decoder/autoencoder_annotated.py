from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import backend as K
import numpy as np

class Autoencoder:
    """
    Represents a Deep Convolutional Autoencoder architecture with mirrored encoder and decoder components.
    
    Attributes:
    - input_shape: Tuple representing the shape of the input (height, width, channels).
    - conv_filters: List of integers, specifying the number of filters for each convolutional layer.
    - conv_kernels: List of integers, specifying the size of the kernels for each convolutional layer.
    - conv_strides: List of integers, specifying the strides for each convolutional layer.
    - latent_space_dim: Integer representing the dimension of the latent space (compressed representation).
    """

    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):
        """
        Initializes the Autoencoder with its parameters and builds the model.
        
        Args:
        - input_shape: Shape of the input image.
        - conv_filters: List containing the number of filters for each convolutional layer.
        - conv_kernels: List containing the size of each kernel for the convolutional layers.
        - conv_strides: List containing the stride for each convolutional layer.
        - latent_space_dim: Dimensionality of the latent space (i.e., the bottleneck layer).
        """
        # Assign input parameters to instance variables
        self.input_shape = input_shape  # [28, 28, 1] for grayscale images of size 28x28
        self.conv_filters = conv_filters  # [32, 64, 64, 64], number of filters in each conv layer
        self.conv_kernels = conv_kernels  # [3, 3, 3, 3], kernel size for each conv layer
        self.conv_strides = conv_strides  # [1, 2, 2, 1], stride for each conv layer
        self.latent_space_dim = latent_space_dim  # 2, dimension of the latent space

        # Initialize model components
        self.encoder = None
        self.decoder = None
        self.model = None

        # Number of convolutional layers
        self._num_conv_layers = len(conv_filters)
        # Shape before the bottleneck (needed for decoder)
        self._shape_before_bottleneck = None
        # Input tensor
        self._model_input = None

        # Build the model architecture
        self._build()

    def summary(self):
        """Prints the summary of the encoder, decoder, and autoencoder models."""
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate=0.0001):
        """
        Compiles the autoencoder model with Mean Squared Error loss and Adam optimizer.
        
        Args:
        - learning_rate: Learning rate for the optimizer.
        """
        optimizer = Adam(learning_rate=learning_rate)
        mse_loss = MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss=mse_loss)

    def train(self, x_train, batch_size, num_epochs):
        """
        Trains the autoencoder model.
        
        Args:
        - x_train: Training data.
        - batch_size: Size of each batch.
        - num_epochs: Number of epochs to train for.
        """
        self.model.fit(
            x_train, x_train,  # Autoencoders aim to reproduce their input
            batch_size=batch_size,
            epochs=num_epochs,
            shuffle=True
        )

    def _build(self):
        """Builds the encoder, decoder, and full autoencoder model."""
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_autoencoder(self):
        """Connects the encoder and decoder to create the full autoencoder model."""
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))  # Decoder takes encoder's output as input
        self.model = Model(model_input, model_output, name="autoencoder")

    def _build_decoder(self):
        """Builds the decoder component of the autoencoder."""
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    def _add_decoder_input(self):
        """Defines the input layer for the decoder."""
        return Input(shape=self.latent_space_dim, name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        """Adds a dense layer to the decoder."""
        num_neurons = np.prod(self._shape_before_bottleneck)  # Calculate total number of neurons needed
        dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        """Reshapes the output of the dense layer to the required shape before upsampling."""
        return Reshape(self._shape_before_bottleneck)(dense_layer)

    def _add_conv_transpose_layers(self, x):
        """Adds all transpose convolutional layers to the decoder."""
        for layer_index in reversed(range(1, self._num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_index, x)
        return x

    def _add_conv_transpose_layer(self, layer_index, x):
        """Adds a single transpose convolutional layer."""
        layer_num = self._num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_num}"
        )
        x = conv_transpose_layer(x)
        x = ReLU(name=f"decoder_relu_{layer_num}")(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
        return x

    def _add_decoder_output(self, x):
        """Adds the final output layer to the decoder."""
        conv_transpose_layer = Conv2DTranspose(
            filters=1,  # Single channel for grayscale image output
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )
        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name="sigmoid_layer")(x)  # Sigmoid activation for output between 0 and 1
        return output_layer

    def _build_encoder(self):
        """Builds the encoder component of the autoencoder."""
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="encoder")

    def _add_encoder_input(self):
        """Defines the input layer for the encoder."""
        return Input(shape=self.input_shape, name="encoder_input")

    def _add_conv_layers(self, encoder_input):
        """Adds all convolutional layers to the encoder."""
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        """Adds a single convolutional layer."""
        layer_number = layer_index + 1
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"encoder_conv_layer_{layer_number}"
        )
        x = conv_layer(x)
        x = ReLU(name=f"encoder_relu_{layer_number}")(x)
        x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
        return x

    def _add_bottleneck(self, x):
        """Adds the bottleneck layer to the encoder."""
        self._shape_before_bottleneck = K.int_shape(x)[1:]  # Save the shape before flattening for later use in decoder
        x = Flatten()(x)
        x = Dense(self.latent_space_dim, name="encoder_output")(x)  # Compress representation to latent space
        return x


if __name__ == "__main__":
    # Instantiate the Autoencoder with specified architecture
    autoencoder = Autoencoder(
        input_shape=(28, 28, 1),  # Shape for MNIST dataset
        conv_filters=(32, 64, 64, 64),  # Number of filters in each convolutional layer
        conv_kernels=(3, 3, 3, 3),  # Kernel size for each convolutional layer
        conv_strides=(1, 2, 2, 1),  # Strides for each convolutional layer
        latent_space_dim=2  # Size of the latent space
    )
    # Print the model summaries
    autoencoder.summary()
