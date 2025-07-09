import tkinter as tk
from tkinter import ttk
import numpy as np
import os
import pickle
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import sounddevice as sd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial import KDTree

from autoencoder import VAE
from soundgenerator import SoundGenerator

# Adjust imports and paths according to your project structure
SPECTROGRAMS_PATH = "dataset/spectrograms"
MIN_MAX_VALUES_PATH = "dataset/min_max_values.pkl"
HOP_LENGTH = 256
SAMPLE_RATE = 22050

def load_fsdd(spectrograms_path):
    x_train = []
    X_labels = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            X_label = file_name[0]
            spectrogram = np.load(file_path)
            x_train.append(spectrogram)
            X_labels.append(int(X_label))
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis]
    return x_train, X_labels

def select_images(images, labels, num_images=10000):
    sample_images_index = np.random.choice(range(len(images)), num_images, replace=False)
    sample_images = images[sample_images_index]
    sample_labels = np.array(labels)[sample_images_index]
    return sample_images, sample_labels

class LatentSpaceExplorer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Latent Space Explorer")
        self.geometry("800x600")
        self.create_widgets()
        self.load_model()
        self.plot_latent_space()

    def create_widgets(self):
        # Create a matplotlib figure
        self.figure = plt.Figure(figsize=(6,6))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        # Button to stop audio playback
        self.stop_button = ttk.Button(self, text="Stop Audio", command=self.stop_audio)
        self.stop_button.pack(side=tk.BOTTOM, pady=10)

    def load_model(self):
        # Load the VAE model
        self.vae = VAE.load("model")
        self.sound_generator = SoundGenerator(self.vae, HOP_LENGTH)
        # Load min-max values for spectrogram denormalization
        with open(MIN_MAX_VALUES_PATH, "rb") as f:
            self.min_max_values = pickle.load(f)
        # Load the data
        self.latent_representations, self.sample_labels = self.get_latent_representations()

        # Build a KDTree for nearest neighbor search
        self.kdtree = KDTree(self.latent_2d)

    def get_latent_representations(self):
        # Load data
        x_train, X_labels = load_fsdd(SPECTROGRAMS_PATH)
        num_images = 1000
        sample_images, sample_labels = select_images(x_train, X_labels, num_images)
        _, latent_representations = self.vae.reconstruct(sample_images)

        # Reduce dimensions for plotting
        # Option 1: Using PCA
        pca = PCA(n_components=2)
        self.latent_2d = pca.fit_transform(latent_representations)

        # Option 2: Using t-SNE (comment out PCA and uncomment t-SNE if preferred)
        # tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
        # self.latent_2d = tsne.fit_transform(latent_representations)

        self.latent_representations_full = latent_representations
        return latent_representations, sample_labels

    def plot_latent_space(self):
        scatter = self.ax.scatter(
            self.latent_2d[:, 0],
            self.latent_2d[:, 1],
            c=self.sample_labels,
            cmap="tab10",
            alpha=0.5,
            s=2
        )
        self.ax.set_title("Latent Space")
        self.figure.colorbar(scatter, ax=self.ax)
        self.canvas.draw()

    def on_click(self, event):
        if event.inaxes == self.ax:
            x = event.xdata
            y = event.ydata
            print(f"Clicked at ({x}, {y})")
            # Find the nearest latent vector
            self.generate_audio([x, y])

    def generate_audio(self, clicked_point):
        # Find the nearest neighbor in the latent space
        distance, index = self.kdtree.query(clicked_point)
        latent_vector = self.latent_representations_full[index].reshape(1, -1)
        # Generate spectrogram from latent vector
        generated_spectrogram = self.vae.decoder.predict(latent_vector)
        # Since the generator expects min_max_values, we'll use average min and max
        avg_min = np.mean([v["min"] for v in self.min_max_values.values()])
        avg_max = np.mean([v["max"] for v in self.min_max_values.values()])
        min_max_values = [{"min": avg_min, "max": avg_max}]
        # Convert spectrogram to audio
        signals = self.sound_generator.convert_spectrograms_to_audio(
            generated_spectrogram, min_max_values
        )
        signal = signals[0]
        # Play the audio
        self.play_audio(signal)

    def play_audio(self, signal):
        self.stop_audio()  # Stop any existing playback
        sd.play(signal, samplerate=SAMPLE_RATE)

    def stop_audio(self):
        sd.stop()

if __name__ == "__main__":
    app = LatentSpaceExplorer()
    app.mainloop()
