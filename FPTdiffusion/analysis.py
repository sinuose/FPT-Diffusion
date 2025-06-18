import numpy as np
import pandas as pd

from scipy.optimize import curve_fit
from tqdm import tqdm

def gaussian_2d(xy, A, x0, y0, sigma_x, sigma_y, theta, offset):
    x, y = xy
    x0 = float(x0)
    y0 = float(y0)
    a = (np.cos(theta)**2) / (2*sigma_x**2) + (np.sin(theta)**2) / (2*sigma_y**2)
    b = -np.sin(2*theta) / (4*sigma_x**2) + np.sin(2*theta) / (4*sigma_y**2)
    c = (np.sin(theta)**2) / (2*sigma_x**2) + (np.cos(theta)**2) / (2*sigma_y**2)
    g = A * np.exp(-(a*(x - x0)**2 + 2*b*(x - x0)*(y - y0) + c*(y - y0)**2)) + offset
    return g.ravel()

# analysis function that takes in results from "batch" or "locate" and spits out dope ass analysis
def fit_function(data, function):
    """
    Fits a 2D Gaussian to the given image patch.
    
    Parameters:
        data: 2D NumPy array (image patch)
        
    Returns:
        popt: Best-fit parameters (A, x0, y0, sigma_x, sigma_y, theta, offset)
        pcov: Covariance matrix of the fit parameters
    """
    y_size, x_size = data.shape
    x = np.arange(x_size)
    y = np.arange(y_size)
    x, y = np.meshgrid(x, y)

    # Initial guess
    A_init = np.max(data) - np.min(data)
    x0_init = x_size / 2
    y0_init = y_size / 2
    sigma_x_init = sigma_y_init = min(x_size, y_size) / 4
    theta_init = 0
    offset_init = np.min(data)

    initial_guess = (A_init, x0_init, y0_init, sigma_x_init, sigma_y_init, theta_init, offset_init)

    try:
        popt, pcov = curve_fit(
            function,
            (x, y),
            data.ravel(),
            p0=initial_guess,
            bounds=(
                [0, 0, 0, 0, 0, -np.pi, -np.inf],   # lower bounds
                [np.inf, x_size, y_size, x_size, y_size, np.pi, np.inf]  # upper bounds
            )
        )
        return popt, pcov
    except RuntimeError:
        return None, None
    
# one function for each feature so its modular
def subpixelGaussian(img, results, box_size=7, function=gaussian_2d):
    """
    Refines particle positions in 'results' using a 2D Gaussian fit.

    Parameters:
        img (ndarray): 2D (single frame) or 3D (video: T, H, W) image.
        results (pd.DataFrame): Must contain ['y', 'x', 'frame'] columns.
        function (callable): Gaussian model function (default: gaussian_2d).
        box_size (int): Size of square region around particle (must be odd).

    Returns:
        pd.DataFrame: Results with refined x/y and fit diagnostics.
    """
    assert box_size % 2 == 1, "box_size must be odd"
    half_box = box_size // 2

    # Copy the results to avoid modifying in-place
    refined = results.copy()
    refined['x_fit'] = np.nan
    refined['y_fit'] = np.nan
    refined['amplitude'] = np.nan
    refined['sigma_x'] = np.nan
    refined['sigma_y'] = np.nan
    refined['theta'] = np.nan
    refined['offset'] = np.nan
    refined['fit_success'] = False

    for i, row in tqdm(refined.iterrows()):
        x, y = row['x'], row['y']
        frame_idx = int(row['frame']) if np.ndim(img) == 3 else None

        # Select the correct image slice
        image = img[frame_idx] if np.ndim(img) == 3 else img

        # Integer center of the bounding box
        x0, y0 = int(round(x)), int(round(y))

        # Ensure bounding box is within image bounds
        y_start = max(y0 - half_box, 0)
        y_end = min(y0 + half_box + 1, image.shape[0])
        x_start = max(x0 - half_box, 0)
        x_end = min(x0 + half_box + 1, image.shape[1])

        patch = image[y_start:y_end, x_start:x_end]

        # Skip fitting if patch is too small
        if patch.shape[0] < 3 or patch.shape[1] < 3:
            continue

        popt, _ = fit_function(patch, function)
        if popt is not None:
            A, x_fit, y_fit, sigma_x, sigma_y, theta, offset = popt
            refined.at[i, 'x_fit'] = x_start + x_fit
            refined.at[i, 'y_fit'] = y_start + y_fit
            refined.at[i, 'amplitude'] = A
            refined.at[i, 'sigma_x'] = sigma_x
            refined.at[i, 'sigma_y'] = sigma_y
            refined.at[i, 'theta'] = theta
            refined.at[i, 'offset'] = offset
            refined.at[i, 'fit_success'] = True

    return refined


def bgdSignal(video, refined, sigma_multiplier=2):
    """
    Estimate background mean and noise from video using refined Gaussian fit results.

    Parameters:
        video (list or ndarray): List/array of 2D images (frames)
        refined (DataFrame): Output from subpixelGaussian() with x_fit, y_fit, sigma_x, sigma_y
        sigma_multiplier (float): Number of sigmas to mask out each Gaussian

    Returns:
        bgd_stats (DataFrame): One row per frame with mean and std of background pixels
    """
    H, W = video[0].shape
    bgd_stats = []

    for frame_num, frame in enumerate(video):
        frame_data = refined[(refined['frame'] == frame_num) & (refined['fit_success'])]

        # Start with all pixels valid
        mask = np.ones((H, W), dtype=bool)

        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

        for _, row in frame_data.iterrows():
            x0, y0 = row['x_fit'], row['y_fit']
            sx, sy = row['sigma_x'], row['sigma_y']
            if any(np.isnan([x0, y0, sx, sy])):
                continue

            # Convert from image coords to pixel grid coords
            # Assuming your image spans [-10,10] in x and y
            X = np.linspace(-10, 10, W)
            Y = np.linspace(-10, 10, H)
            x_pixel = np.interp(x0, X, np.arange(W))
            y_pixel = np.interp(y0, Y, np.arange(H))

            # Construct elliptical mask
            dx = xx - x_pixel
            dy = yy - y_pixel
            region_mask = ((dx / (sigma_multiplier * sx * W / 20))**2 +
                           (dy / (sigma_multiplier * sy * H / 20))**2) <= 1
            mask &= ~region_mask  # remove area within Gaussian

        background_pixels = frame[mask]
        bg_mean = np.mean(background_pixels)
        bg_std = np.std(background_pixels)

        bgd_stats.append({
            'frame': frame_num,
            'bg_mean': bg_mean,
            'bg_std': bg_std,
            'num_pixels_sampled': np.sum(mask)
        })

    return pd.DataFrame(bgd_stats)