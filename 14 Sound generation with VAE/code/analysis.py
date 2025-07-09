import numpy as np
import matplotlib.pyplot as plt
import os
from autoencoder import VAE

def load_fsdd(spectrograms_path):
    x_train = []
    X_labels = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            X_label = file_name[0]
            spectrogram = np.load(file_path) # (n_bins, n_frames, 1)
            x_train.append(spectrogram)
            X_labels.append(int(X_label))
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis] # -> (3000, 256, 64, 1)
    return x_train, X_labels


def select_images(images, labels, num_images=10):
    sample_images_index = np.random.choice(range(len(images)), num_images)
    sample_images = images[sample_images_index]
    sample_labels = np.array(labels)[sample_images_index]
    return sample_images, sample_labels



# Function to plot original and reconstructed images side by side
def plot_reconstructed_images(images, reconstructed_images):
    """
    Plot the original images and their reconstructed versions side by side.

    Parameters:
        images (numpy.ndarray): The original images.
        reconstructed_images (numpy.ndarray): The reconstructed images by the autoencoder.
    """
    num_images = len(images)
    
    # Increase the figure size for more space between images
    fig = plt.figure(figsize=(16, 4))  
    
    # Adjust layout to add space between the rows and titles
    plt.subplots_adjust(wspace=0.3, hspace=0.2)  
    
    for i, (image, reconstructed_image) in enumerate(zip(images, reconstructed_images)):
        image = image.squeeze()  # Remove single-dimensional entries from the image
        
        # Plot the original image
        ax = fig.add_subplot(2, num_images, i + 1)
        ax.axis("off")
        ax.imshow(image, cmap="gray_r")
        ax.set_title("Original Image", fontsize=10)  # Smaller font size to avoid crowding
        
        # Plot the reconstructed image
        reconstructed_image = reconstructed_image.squeeze()
        ax = fig.add_subplot(2, num_images, i + num_images + 1)
        ax.axis("off")
        ax.imshow(reconstructed_image, cmap="gray_r")
        ax.set_title("\n Reconstructed Image", fontsize=10)  # Smaller font size for consistency

    # Set an overall title for the figure and adjust its position
    plt.suptitle("Original vs Reconstructed Images", fontsize=18)  # Move the title higher
    # Use tight_layout to automatically adjust subplot spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Leave enough space for the suptitle
    plt.show()

# Function to plot the latent space representation of images
def plot_images_encoded_in_latent_space(latent_representations, sample_labels):
    """
    Plot the encoded images in the latent space, colored by their labels.
    
    Parameters:
        latent_representations (numpy.ndarray): The encoded representations in latent space.
        sample_labels (numpy.ndarray): The labels corresponding to the encoded images.
    """
    plt.figure(figsize=(10, 10))
    plt.scatter(latent_representations[:, 0],  # Latent dimension 1
                latent_representations[:, 1],  # Latent dimension 2
                cmap="rainbow",  # Color map for the points
                c=sample_labels,  # Color the points based on their labels
                alpha=0.5,  # Transparency of points
                s=2)  # Size of points
    plt.colorbar()  # Add a color bar to show the class-color relationship
    plt.title("Latent Space Representation of Test Images", fontsize=16)  # Plot title
    plt.xlabel("Latent Dimension 1")  # Label for x-axis
    plt.ylabel("Latent Dimension 2")  # Label for y-axis
    plt.show()


if __name__ == "__main__":
    SPECTROGRAMS_PATH = "dataset/spectrograms"

    autoencoder = VAE.load("model")
    x_train, X_labels = load_fsdd(SPECTROGRAMS_PATH)


    num_images = 10000
    sample_images, sample_labels = select_images(x_train, X_labels, num_images)
    _, latent_representations = autoencoder.reconstruct(sample_images)
    plot_images_encoded_in_latent_space(latent_representations, sample_labels)






















