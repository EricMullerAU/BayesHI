import numpy as np
from matplotlib import pyplot as plt

def generate_gaussian_spectrum(num_channels=256, max_components=3, noise=0.5):
    x = np.linspace(0, num_channels - 1, num_channels)
    num_components = np.random.randint(1, max_components + 1)
    
    spectrum = np.zeros_like(x)
    center = num_channels / 2
    mean_std = num_channels / 10  # spread of means around the center

    for _ in range(num_components):
        std = np.random.uniform(2, 15)  # standard deviation of Gaussian
        margin = 3 * std

        # Resample mean until it satisfies margin condition
        while True:
            mean = np.random.normal(loc=center, scale=mean_std)
            if margin <= mean <= (num_channels - 1 - margin):
                break

        amp = np.random.uniform(1, 15)  # amplitude
        spectrum += amp * np.exp(-0.5 * ((x - mean) / std) ** 2)

    # Add Gaussian noise
    noise = np.random.uniform(noise, size=num_channels)
    return spectrum + noise


def generate_spectra(num_spectra=100, num_channels=256, max_components=3, noise_std=0.05):
    spectra = np.array([
        generate_gaussian_spectrum(num_channels, max_components, noise_std)
        for _ in range(num_spectra)
    ])
    return spectra.astype(np.float32) # Convert to float32 for PyTorch compatibility

def plot_spectrum(data, predictions, index):
    spectrum = data[index].T
    maximum = np.max(spectrum)

    avg_prediction = np.mean(predictions, axis = 0)
    std_prediction = np.std(predictions, axis = 0) 
    
    plt.figure(figsize=(10,6))
    plt.plot(spectrum)
    plt.xlabel('Channel')
    plt.ylabel(r'$T_B$ [K]')
    plt.text(0, 0.9*maximum, fr'$f_\text{{CNM}} = {avg_prediction[index,0]:.2f} \pm {std_prediction[index,0]:.2f}$')
    plt.text(0, 0.8*maximum, fr'$f_\text{{UNM}} = {avg_prediction[index,1]:.2f} \pm {std_prediction[index,1]:.2f}$')
    plt.text(0, 0.7*maximum, fr'$f_\text{{WNM}} = {avg_prediction[index,2]:.2f} \pm {std_prediction[index,2]:.2f}$')
    plt.text(0, 0.6*maximum, fr'$\mathcal{{R}}_\text{{HI}} = {avg_prediction[index,3]:.2f} \pm {std_prediction[index,3]:.2f}$')
    plt.show()