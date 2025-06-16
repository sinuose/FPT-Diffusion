import numpy as np
import matplotlib.pyplot as plt
from FPTdiffusion import sptObject as sptO
from FPTdiffusion.utils import *
from FPTdiffusion.masks import *
import pandas as pd
from scipy import ndimage


# FUNCTIONS
def _safe_center_of_mass(x, radius, grids):
    normalizer = x.sum()
    if normalizer == 0:  # avoid divide-by-zero errors
        return np.array(radius)
    return np.array([(x * grids[dim]).sum() / normalizer
                    for dim in range(x.ndim)])

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

def percentile_threshold(image, percentile):
    """Find grayscale threshold based on distribution in image."""

    not_black = image[np.nonzero(image)]
    if len(not_black) == 0:
        return np.nan
    return np.percentile(not_black, percentile)

def default_pos_columns(ndim):
    """ Sets the default position column names """
    if ndim < 4:
        return ['z', 'y', 'x'][-ndim:]
    else:
        return list(map(lambda i: 'x' + str(i), range(ndim)))

def convert_to_int(image, dtype='uint8'):
    """Convert the image to integer and normalize if applicable.

    Clips all negative values to 0. Does nothing if the image is already
    of integer type.

    Parameters
    ----------
    image : ndarray
    dtype : numpy dtype
        dtype to convert to. If the image is already of integer type, this
        argument is ignored. Must be integer-subdtype. Default 'uint8'.

    Returns
    -------
    tuple of (scale_factor, image)
    """
    if np.issubdtype(image.dtype, np.integer):
        # Do nothing, image is already of integer type.
        return 1., image
    max_value = np.iinfo(dtype).max
    image_max = image.max()
    if image_max == 0:  # protect against division by zero
        scale_factor = 1.
    else:
        scale_factor = max_value / image_max
    return scale_factor, (scale_factor * image.clip(min=0.)).astype(dtype)

def where_close(pos, separation, intensity=None):
    """
    Identify indices of particles that are closer than `separation` to another.
    If `intensity` is given, retain the brighter particle.
    Else, retain the most top-left particle (smallest coordinate sum).
    """
    if len(pos) == 0:
        return np.array([], dtype=int)

    # Ensure pos is a NumPy array
    pos = pos.values if isinstance(pos, pd.DataFrame) else np.asarray(pos)
    ndim = pos.shape[1]
    separation = validate_tuple(separation, ndim)

    if any(s == 0 for s in separation):
        return np.array([], dtype=int)

    # Normalize positions based on separation
    pos_rescaled = pos / np.array(separation)

    N = len(pos_rescaled)
    to_drop = set()

    for i in range(N):
        if i in to_drop:
            continue

        diff = pos_rescaled[i+1:] - pos_rescaled[i]
        dists_sq = np.sum(diff**2, axis=1)
        close_mask = dists_sq < (1 - 1e-7)**2
        close_idx = np.where(close_mask)[0] + i + 1

        for j in close_idx:
            if j in to_drop or i in to_drop:
                continue

            if intensity is not None:
                int_i = intensity[i]
                int_j = intensity[j]
                if int_i < int_j:
                    to_drop.add(i)
                elif int_i > int_j:
                    to_drop.add(j)
                else:
                    sum_i = np.sum(pos_rescaled[i])
                    sum_j = np.sum(pos_rescaled[j])
                    to_drop.add(i if sum_i > sum_j else j)
            else:
                sum_i = np.sum(pos_rescaled[i])
                sum_j = np.sum(pos_rescaled[j])
                to_drop.add(i if sum_i > sum_j else j)

    return np.array(sorted(to_drop), dtype=int)

def drop_close(pos, separation, intensity=None):
    """ Removes features that are closer than separation from other features.
    When intensity is given, the one with the lowest intensity is dropped:
    else the most topleft is dropped (to avoid randomness)"""
    to_drop = where_close(pos, separation, intensity)
    return np.delete(pos, to_drop, axis=0)

def grey_dilation(image, separation, percentile=64, margin=None, precise=True):
    """Find local maxima whose brightness is above a given percentile.

    Parameters
    ----------
    image : ndarray
        For best performance, provide an integer-type array. If the type is not
        of integer-type, the image will be normalized and coerced to uint8.
    separation : number or tuple of numbers
        Minimum separation between maxima. See precise for more information.
    percentile : float in range of [0,100], optional
        Features must have a peak brighter than pixels in this percentile.
        This helps eliminate spurious peaks. Default 64.
    margin : integer or tuple of integers, optional
        Zone of exclusion at edges of image. Default is ``separation / 2``.
    precise : boolean, optional
        Determines whether there will be an extra filtering step (``drop_close``)
        discarding features that are too close. Degrades performance.
        Because of the square kernel used, too many features are returned when
        precise=False. Default True.

    See Also
    --------
    drop_close : removes features that are too close to brighter features
    grey_dilation_legacy : local maxima finding routine used until trackpy v0.3
    """
    # convert to integer. does nothing if image is already of integer type
    factor, image = convert_to_int(image, dtype=np.uint8)

    ndim = image.ndim
    separation = validate_tuple(separation, ndim)
    if margin is None:
        margin = tuple([int(s / 2) for s in separation])

    # Compute a threshold based on percentile.
    threshold = percentile_threshold(image, percentile)

    if np.isnan(threshold):
        #warnings.warn("Image is completely black.", UserWarning)
        return np.empty((0, ndim))

    # Find the largest box that fits inside the ellipse given by separation
    size = [int(2 * s / np.sqrt(ndim)) for s in separation]

    # The intersection of the image with its dilation gives local maxima.
    dilation = ndimage.grey_dilation(image, size, mode='constant')
    maxima = (image == dilation) & (image > threshold)
    if np.sum(maxima) == 0:
        #warnings.warn("Image contains no local maxima.", UserWarning)
        return np.empty((0, ndim))

    pos = np.vstack(np.where(maxima)).T

    # Do not accept peaks near the edges.
    shape = np.array(image.shape)
    near_edge = np.any((pos < margin) | (pos > (shape - margin - 1)), 1)
    pos = pos[~near_edge]

    if len(pos) == 0:
        #warnings.warn("All local maxima were in the margins.", UserWarning)
        return np.empty((0, ndim))

    # Remove local maxima that are too close to each other
    if precise:
        pos = drop_close(pos, separation, image[maxima][~near_edge])

    return pos

def _refine(raw_image, image, radius, coords, max_iterations,
            shift_thresh, characterize, walkthrough):
    if not np.issubdtype(coords.dtype, np.integer):
        raise ValueError('The coords array should be of integer datatype')
    ndim = image.ndim
    isotropic = np.all(radius[1:] == radius[:-1])
    mask = binary_mask(radius, ndim).astype(np.uint8)

    # Declare arrays that we will fill iteratively through loop.
    N = coords.shape[0]
    final_coords = np.empty_like(coords, dtype=np.float64)
    mass = np.empty(N, dtype=np.float64)
    raw_mass = np.empty(N, dtype=np.float64)
    if characterize:
        if isotropic:
            Rg = np.empty(N, dtype=np.float64)
        else:
            Rg = np.empty((N, len(radius)), dtype=np.float64)
        ecc = np.empty(N, dtype=np.float64)
        signal = np.empty(N, dtype=np.float64)

    ogrid = np.ogrid[[slice(0, i) for i in mask.shape]]  # for center of mass
    ogrid = [g.astype(float) for g in ogrid]

    for feat, coord in enumerate(coords):
        for iteration in range(max_iterations):
            # Define the circular neighborhood of (x, y).
            rect = tuple([slice(c - r, c + r + 1)
                          for c, r in zip(coord, radius)])
            neighborhood = mask * image[rect]
            cm_n = _safe_center_of_mass(neighborhood, radius, ogrid)
            cm_i = cm_n - radius + coord  # image coords

            off_center = cm_n - radius

            #logger.debug('off_center: %f', off_center)
            if np.all(np.abs(off_center) < shift_thresh):
                break  # Accurate enough.
            # If we're off by more than half a pixel in any direction, move..
            coord[off_center > shift_thresh] += 1
            coord[off_center < -shift_thresh] -= 1
            # Don't move outside the image!
            upper_bound = np.array(image.shape) - 1 - radius
            coord = np.clip(coord, radius, upper_bound).astype(int)

        # stick to yx column order
        final_coords[feat] = cm_i

        if walkthrough:
            import matplotlib.pyplot as plt
            plt.imshow(neighborhood)

        # Characterize the neighborhood of our final centroid.
        mass[feat] = neighborhood.sum()
        if not characterize:
            continue  # short-circuit loop
        if isotropic:
            Rg[feat] = np.sqrt(np.sum(r_squared_mask(radius, ndim) *
                                      neighborhood) / mass[feat])
        else:
            Rg[feat] = np.sqrt(ndim * np.sum(x_squared_masks(radius, ndim) *
                                             neighborhood,
                                             axis=tuple(range(1, ndim + 1))) /
                               mass[feat])
        # I only know how to measure eccentricity in 2D.
        if ndim == 2:
            ecc[feat] = np.sqrt(np.sum(neighborhood*cosmask(radius))**2 +
                                np.sum(neighborhood*sinmask(radius))**2)
            ecc[feat] /= (mass[feat] - neighborhood[radius] + 1e-6)
        else:
            ecc[feat] = np.nan
        signal[feat] = neighborhood.max()  # based on bandpassed image
        raw_neighborhood = mask * raw_image[rect]
        raw_mass[feat] = raw_neighborhood.sum()  # based on raw image

    if not characterize:
        return np.column_stack([final_coords, mass])
    else:
        return np.column_stack([final_coords, mass, Rg, ecc, signal, raw_mass])


def refine_com_arr(raw_image, image, radius, coords, max_iterations=10,shift_thresh=0.6, characterize=True,
                   walkthrough=False):
    """Refine coordinates and return a numpy array instead of a DataFrame.

    See also
    --------
    refine_com
    """
    if raw_image.ndim != coords.shape[1]:
        raise ValueError("The image has a different number of dimensions than "
                         "the coordinate array.")

    # ensure that radius is tuple of integers, for direct calls to refine_com_arr()
    radius = validate_tuple(radius, image.ndim)

    # In here, coord is an integer. Make a copy, will not modify inplace.
    coords = np.round(coords).astype(int)

    results = _refine(raw_image, image, radius, coords, max_iterations,
                          shift_thresh, characterize, walkthrough)

    return results

def refine_com(raw_image, image, radius, coords, max_iterations=10,
               engine='auto', shift_thresh=0.6, characterize=True,
               pos_columns=None):
    
    if isinstance(coords, pd.DataFrame):
        if pos_columns is None:
            pos_columns = guess_pos_columns(coords)
        index = coords.index
        coords = coords[pos_columns].values
    else:
        index = None

    radius = validate_tuple(radius, image.ndim)

    if pos_columns is None:
        pos_columns = default_pos_columns(image.ndim)
    columns = pos_columns + ['mass']
    if characterize:
        isotropic = radius[1:] == radius[:-1]
        columns += default_size_columns(image.ndim, isotropic) + \
            ['ecc', 'signal', 'raw_mass']

    if len(coords) == 0:
        return pd.DataFrame(columns=columns)

    refined = refine_com_arr(raw_image, image, radius, coords,
                             max_iterations=max_iterations, shift_thresh=shift_thresh,
                             characterize=characterize)

    return pd.DataFrame(refined, columns=columns, index=index)

def locate(raw_image, diameter, minmass=None, maxsize=None, separation=None,
           noise_size=1, smoothing_size=None, threshold=None, invert=False,
           percentile=64, topn=None, preprocess=True, max_iterations=10,
           filter_before=None, filter_after=None,
           characterize=True):

    # Validate parameters and set defaults.
    raw_image = np.squeeze(raw_image)
    shape = raw_image.shape
    ndim = len(shape)

    # Crocker-Grier assumes rough particle size
    diameter = validate_tuple(diameter, ndim)
    diameter = tuple([int(x) for x in diameter])
    if not np.all([x & 1 for x in diameter]):
        raise ValueError("Feature diameter must be an odd integer. Round up.")
    
    radius = tuple([x//2 for x in diameter])

    isotropic = np.all(radius[1:] == radius[:-1])
    if (not isotropic) and (maxsize is not None):
        raise ValueError("Filtering by size is not available for anisotropic "
                         "features.")

    is_float_image = not np.issubdtype(raw_image.dtype, np.integer)

    if separation is None:
        separation = tuple([x + 1 for x in diameter])

    if smoothing_size is None:
        smoothing_size = diameter

    if minmass is None:
        minmass = 0

    if threshold is None:
        if is_float_image:
            threshold = 1/255.
        else:
            threshold = 1

    # Invert the image if necessary
    if invert:
        print("fix later")
        #raw_image = invert_image(raw_image)

    # Determine `image`: the image to find the local maxima on.
    if preprocess:
        print('fix later')
        image = raw_image#bandpass(raw_image, noise_size, smoothing_size, threshold)
    else:
        image = raw_image

    # For optimal performance, performance, coerce the image dtype to integer.
    if is_float_image:  # For float images, assume bitdepth of 8.
        dtype = np.uint8
    else:   # For integer images, take original dtype
        dtype = raw_image.dtype

    # Normalize_to_int does nothing if image is already of integer type.
    scale_factor, image = convert_to_int(image, dtype)

    pos_columns = default_pos_columns(image.ndim)

    # Find local maxima.
    # Define zone of exclusion at edges of image, avoiding
    #   - Features with incomplete image data ("radius")
    #   - Extended particles that cannot be explored during subpixel
    #       refinement ("separation")
    #   - Invalid output of the bandpass step ("smoothing_size")
    margin = tuple([max(rad, sep // 2 - 1, sm // 2) for (rad, sep, sm) in
                    zip(radius, separation, smoothing_size)])
    
    # Find features with minimum separation distance of `separation`. This
    # excludes detection of small features close to large, bright features
    # using the `maxsize` argument.
    coords = grey_dilation(image, separation, percentile, margin, precise=False)

    # Refine their locations and characterize mass, size, etc.
    # need this function 
    refined_coords = refine_com(raw_image, image, radius, coords, max_iterations=max_iterations, characterize=characterize)
    # this should spit out a dataframe acctually
    if len(refined_coords) == 0:
        return refined_coords

    # Flat peaks return multiple nearby maxima. Eliminate duplicates.
    if np.all(np.greater(separation, 0)):
        to_drop = where_close(refined_coords[pos_columns], separation,
                              refined_coords['mass'])
        refined_coords.drop(to_drop, axis=0, inplace=True)
        refined_coords.reset_index(drop=True, inplace=True)

    # mass and signal values has to be corrected due to the rescaling
    # raw_mass was obtained from raw image; size and ecc are scale-independent
    refined_coords['mass'] /= scale_factor
    if 'signal' in refined_coords:
        refined_coords['signal'] /= scale_factor

    # Filter on mass and size, if set.
    condition = refined_coords['mass'] > minmass
    if maxsize is not None:
        condition &= refined_coords['size'] < maxsize
    if not condition.all():  # apply the filter
        # making a copy to avoid SettingWithCopyWarning
        refined_coords = refined_coords.loc[condition].copy()

    if len(refined_coords) == 0:
        #warnings.warn("No maxima survived mass- and size-based filtering. "
        return refined_coords

    if topn is not None and len(refined_coords) > topn:
        # go through numpy for easy pandas backwards compatibility
        mass = refined_coords['mass'].values
        if topn == 1:
            # special case for high performance and correct shape
            refined_coords = refined_coords.iloc[[np.argmax(mass)]]
        else:
            refined_coords = refined_coords.iloc[np.argsort(mass)[-topn:]]

    # If this is a pims Frame object, it has a frame number.
    # Tag it on; this is helpful for parallelization.
    if hasattr(raw_image, 'frame_no') and raw_image.frame_no is not None:
        refined_coords['frame'] = int(raw_image.frame_no)
    return refined_coords




# Main

# make some bullshit image
img = np.zeros((512,512))
nump = 5                            # number of particles.
gen = np.random.default_rng(2)

def Gaussian(x,y,x0=0,y0=0, sigma=0.1):
    return np.exp(- ((x-x0)**2 + (y-y0)**2)/(2*sigma))

xrange = np.linspace(-10, 10,512)
yrange = np.linspace(-10, 10,512)

xv, yv = np.meshgrid(xrange, yrange)

for i in range(nump):
    num1 = gen.uniform(-10, 10)
    num2 = gen.uniform(-10, 10) #np.sqrt(10-num1)
    img+= Gaussian(xv,yv, num1, num2)
    

img = add_heavy_noise(img)

# location
coords = locate(img, 7, 20)

x = coords['x']
y = coords['y']

plt.imshow(img)
plt.scatter(x,y, color='red')
plt.show()


