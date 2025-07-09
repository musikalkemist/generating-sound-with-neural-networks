import os
import pickle
import numpy as np
import soundfile as sf
from pathlib import Path
from soundgenerator import SoundGenerator
from autoencoder import VAE
from train import SPECTROGRAMS_PATH

HOP_LENGTH = 256
SAVE_DIR_ORIGINAL = "samples/original/"
SAVE_DIR_GENERATED = "samples/generated/"
MIN_MAX_VALUES_PATH = "dataset/min_max_values.pkl"

def find_max_shape(spectrograms_path):
    max_rows, max_cols = 0, 0
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)
            rows, cols = spectrogram.shape
            max_rows = max(max_rows, rows)
            max_cols = max(max_cols, cols)
    return max_rows, max_cols

def load_fsdd(spectrograms_path):
    x_train = []
    file_paths = []
    max_rows, max_cols = find_max_shape(spectrograms_path)

    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)
            # Pad or trim spectrogram
            padded_spectrogram = np.zeros((max_rows, max_cols))
            rows, cols = spectrogram.shape
            padded_spectrogram[:rows, :cols] = spectrogram[:max_rows, :max_cols]
            x_train.append(padded_spectrogram)
            file_paths.append(file_path)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis]  # Add channel dimension
    return x_train, file_paths

def select_spectrograms(spectrograms, file_paths, min_max_values, num_spectrograms=2):
    sampled_indexes = np.random.choice(range(len(spectrograms)), num_spectrograms, replace=False)
    sampled_spectrograms = spectrograms[sampled_indexes]
    sampled_file_paths = [file_paths[index] for index in sampled_indexes]
    sampled_file_paths = [str(Path(fp).as_posix()) for fp in sampled_file_paths]
    sampled_min_max_values = [min_max_values[fp] for fp in sampled_file_paths]
    return sampled_spectrograms, sampled_min_max_values

def save_signals(signals, save_dir, sample_rate=22050):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, f"{i}.wav")
        sf.write(save_path, signal, sample_rate)

if __name__ == "__main__":
    # Initialize sound generator
    vae = VAE.load("model")
    sound_generator = SoundGenerator(vae, HOP_LENGTH)

    # Load spectrograms and min-max values
    with open(MIN_MAX_VALUES_PATH, "rb") as f:
        min_max_values = pickle.load(f)

    specs, file_paths = load_fsdd(SPECTROGRAMS_PATH)

    # Sample spectrograms and min-max values
    sampled_specs, sampled_min_max_values = select_spectrograms(
        specs, file_paths, min_max_values, num_spectrograms=5
    )

    # Generate audio for sampled spectrograms
    signals, _ = sound_generator.generate(sampled_specs, sampled_min_max_values)

    # Convert original spectrogram samples to audio
    original_signals = sound_generator.convert_spectrograms_to_audio(
        sampled_specs, sampled_min_max_values
    )

    # Save audio signals
    save_signals(signals, SAVE_DIR_GENERATED)
    save_signals(original_signals, SAVE_DIR_ORIGINAL)
    print("AUDIO Generated!")
