# When using this package, first import a video as a numpy class. 
# This will be sorted and it will say what options can be done to it so that the process is simple
import numpy as np
import pandas as pd



class sptObject():
    def __init__(self, array):
        # Check that input is a 2D NumPy array
        if not isinstance(array, np.ndarray):
            raise TypeError("Input must be a NumPy array.")
        # distinguish if it is a video or an img. 
        if array.ndim not in [2, 3]:
            raise ValueError("Input array must be 2D (image) or 3D (video).")

        self.ndim = array.ndim

        if self.ndim == 2:
            self.img = array
        elif self.ndim == 3:
            self.video = array
            self.n_frames = array.shape[0]  # Assume time is axis 0

        self.get_statistics()  # Automatically gather stats at init
        
    def get_statistics(self):
        '''
        Computes statistics for either a 2D image or a 3D video.
        Stores the result in self.statistics as a Pandas DataFrame.
        '''
        if self.ndim == 2:
            arr = self.img
            stats = {
                'mean': [np.mean(arr)],
                'std': [np.std(arr)],
                'min': [np.min(arr)],
                'max': [np.max(arr)],
                'shape': [arr.shape],
                'dtype': [arr.dtype],
                'type': ['image']
            }
            self.statistics = pd.DataFrame(stats)

        elif self.ndim == 3:
            arr = self.video
            # Per-frame statistics
            frame_means = np.mean(arr, axis=(1, 2))
            frame_stds = np.std(arr, axis=(1, 2))
            frame_mins = np.min(arr, axis=(1, 2))
            frame_maxs = np.max(arr, axis=(1, 2))

            stats = {
                'frame': list(range(self.n_frames)),
                'mean': frame_means,
                'std': frame_stds,
                'min': frame_mins,
                'max': frame_maxs
            }
            self.statistics = pd.DataFrame(stats)
            self.statistics['shape'] = [arr.shape[1:]] * self.n_frames
            self.statistics['dtype'] = [arr.dtype] * self.n_frames
            self.statistics['type'] = ['video'] * self.n_frames

        return self.statistics