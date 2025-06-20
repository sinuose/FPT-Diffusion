# When using this package, first import a video as a numpy class. 
# This will be sorted and it will say what options can be done to it so that the process is simple
import numpy as np
import pandas as pd

from .spt import *
from .mpt import *
from .analysis import *


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

        self.params = {
            'diameter': (7, 7),
            'minmass': 20,
            'separation': (15,15),
            'smoothing_size': None,
            'threshold': None,
            'percentile': 64,
            'topn': None,
            'preprocess': False,
            'characterize': False
        }
        print("Initialization Parameters: ", self.params)
        
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
    
    def StandardSPT(self, refine=True, sigbgd=True):
        # if image
        if self.ndim == 2:
            # STEP 1. GET ALL THE POSITIONS
            self.result_df = batch(self.img, self.params)
            # STEP 2. Get all the refined positions
            if refine:
                self.result_df = subpixelGaussian(self.img, self.result_df, 15)
            # STEP 3. obtain background information
            if sigbgd:
                self.sigbgd_df = bgdSignal(self.img, self.result_df, 1)

        # if video
        elif self.ndim == 3:
            # STEP 1. GET ALL THE POSITIONS
            self.result_df = batch(self.video, self.params)
            # STEP 2. Get all the refined positions
            if refine:
                self.result_df = subpixelGaussian(self.video, self.result_df, 15)
            # STEP 3. obtain background information
            if sigbgd:
                self.sigbgd_df = bgdSignal(self.video, self.result_df, 1)

    def LinkParticles(self, max_distance = 30, birth_death_cost = 100):
        # only requiredment is a results_df with frame indicies.
        if set(['frame','x', 'y']).issubset(self.result_df.columns):
            pass
        else:
            raise ValueError("Video must be analyzed and contain frame and trajectory information.")

        # now, run the custom grand cannonical hungarian algorithm
        self.result_df = gcHungarian(self.result_df, max_distance, birth_death_cost )

    def GetSptResults(self):
        try:
            return self.result_df
        except Exception as e:
            print(e)
    
    def GetSigResults(self):
        try:
            return self.sigbgd_df
        except Exception as e:
            print(e)