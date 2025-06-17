import numpy as np
from .utils import *
from scipy import ndimage

def percentile_threshold(image, percentile):
    """Find grayscale threshold based on distribution in image."""

    not_black = image[np.nonzero(image)]
    if len(not_black) == 0:
        return np.nan
    return np.percentile(not_black, percentile)


