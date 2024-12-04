import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def visualize_samples(X, save_dir='samples_visualization'):
    backward = 50
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define colors for each segment
    colors = ['b', 'g', 'r', 'c', 'm']

    # Loop through each sample in X
    for i, sample in enumerate(X):
        plt.figure(figsize=(12, 7))

        # Plot each segment with its respective color
        for j, color in enumerate(colors):
            plt.plot(range(j * backward, (j + 1) * backward), sample[j * backward:(j + 1) * backward], color=color,
                     label=f'Segment {j + 1}')

        plt.title(f'Sample {i}')
        plt.xlabel('Index')
        plt.ylabel('Value')
        #plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)

        # Save the figure
        plt.savefig(os.path.join(save_dir, f'sample_{i}.png'))
        plt.close()  # Close the figure to free up memory


X = np.load('X.npy')
visualize_samples(X)