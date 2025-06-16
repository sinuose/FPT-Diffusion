# This file will have functions used for tracking for 2D images.
import pywt
# What are the best algorithimns for particle trackign single particles?
# need a quick and easy local maxima search
# to also make this robust, we can assume that the input images are just 2D numpy arrays.

def sptCheck(array):
    # Check that input is a 2D NumPy array
    if not isinstance(array, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    # must be 2d 
    if array.ndim not in [2]:
        raise ValueError("Input array must be 2D (image)")

def lm_wavelet(array):
    sptCheck(array)

    # now search for local maxima using wavelts

     

# then need to refine to get sub pixel resolution




