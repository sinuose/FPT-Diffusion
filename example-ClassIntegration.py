import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from FPTdiffusion.sptObject import *

# FUNCTIONS
def add_heavy_noise(image, noise_level=0.5):
    """
    Adds strong Gaussian noise to a grayscale image.

    Parameters:
    - image: 2D NumPy array (grayscale image)
    - noise_level: standard deviation of noise relative to max(image). Default = 0.5

    Returns:
    - noisy_image: 2D NumPy array with noise added
    """
    image = image.astype(np.float32)
    noise_std = noise_level * np.max(image)
    noise = np.random.normal(0, noise_std, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 1)  # Keep in displayable range
    return noisy_image

# Main
img = np.zeros((512,512))
nump = 3                         # number of particles.
gen = np.random.default_rng()

def Gaussian(x, y, x0=0, y0=0, sigma=0.1):
    return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

xrange = np.linspace(-10, 10, 512)
yrange = np.linspace(-10, 10, 512)
xv, yv = np.meshgrid(xrange, yrange)

# Initialize particle positions (x, y) in [-8, 8]
positions = gen.uniform(-8, 8, size=(nump, 2))

video = []
for j in range(15):
    img = np.zeros((512, 512))
    
    for i in range(nump):
        # Apply random walk step to the position
        step = gen.normal(0, 0.5, size=2)  # small drift
        positions[i] += step
        
        # Keep positions within bounds [-10, 10]
        positions[i] = np.clip(positions[i], -10, 10)
        
        # Unpack and render Gaussian
        x0, y0 = positions[i]
        img += Gaussian(xv, yv, x0, y0, sigma=0.3)
        
    img = add_heavy_noise(img)
    video.append(img)

# Running the batch function
spt = sptObject(np.array(video))

spt.StandardSPT()
spt.LinkParticles(100,2000)

refined_df = spt.GetSptResults()

sigbgd_df = spt.GetSigResults()



print(refined_df)
print(sigbgd_df)

#PLOTTING 

from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation

def plot_gaussian_movie(video, refined, interval=100, save_path=None):
    """
    Creates an animation of a video with 2D Gaussian ellipses overlayed.

    Parameters:
        video (ndarray): 3D array (T, H, W)
        refined (pd.DataFrame): Output from subpixelGaussian() with 'x_fit', 'y_fit', 'sigma_x', 'sigma_y', 'theta'
        interval (int): Delay between frames in milliseconds
        save_path (str): If provided, saves the animation (e.g., 'movie.mp4')
    """
    T, H, W = np.shape(video)
    fig, ax = plt.subplots()
    im = ax.imshow(video[0], cmap='gray', vmin=np.min(video), vmax=np.max(video))
    ellipses = []

    def draw_ellipses(frame_num):
        ax.clear()
        ax.imshow(video[frame_num], cmap='gray', vmin=np.min(video), vmax=np.max(video))
        frame_data = refined[(refined['frame'] == frame_num) & (refined['fit_success'])]
        for _, row in frame_data.iterrows():
            e = Ellipse(
                xy=(row['x_fit'], row['y_fit']),
                width=2*row['sigma_x'],
                height=2*row['sigma_y'],
                angle=np.degrees(row['theta']),
                edgecolor='red',
                facecolor='none',
                linewidth=1
            )
            ax.add_patch(e)
            ax.text(row['x_fit'], row['y_fit'], str(row['particle']), color='lightsteelblue')
        ax.set_title(f"Frame {frame_num}")
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)  # Flip y-axis for image convention

    anim = FuncAnimation(fig, draw_ellipses, frames=T, interval=interval)

    if save_path:
        anim.save(save_path, fps=1000//interval, dpi=150)
    else:
        plt.show()

# Example usage
plot_gaussian_movie(video, refined_df, interval=250)


