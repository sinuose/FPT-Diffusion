import numpy as np

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


def validate_tuple(value, ndim):
    if not hasattr(value, '__iter__'):
        return (value,) * ndim
    if len(value) == ndim:
        return tuple(value)
    raise ValueError("List length should have same length as image dimensions.")


def guess_pos_columns(f):
    """ Guess the position columns from a given feature DataFrame """
    if 'z' in f:
        pos_columns = ['z', 'y', 'x']
    else:
        pos_columns = ['y', 'x']
    return pos_columns


def default_pos_columns(ndim):
    """ Sets the default position column names """
    if ndim < 4:
        return ['z', 'y', 'x'][-ndim:]
    else:
        return list(map(lambda i: 'x' + str(i), range(ndim)))


def default_size_columns(ndim, isotropic):
    """ Sets the default size column names """
    if isotropic:
        return ['size']
    else:
        return ['size_' + cc for cc in default_pos_columns(ndim)]