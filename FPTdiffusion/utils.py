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