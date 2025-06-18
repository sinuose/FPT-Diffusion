from .spt import *
from tqdm import trange

def batch(frames_array, params):
    all_dfs = []
    for i in trange(np.shape(frames_array)[0]):
        image = frames_array[i]
        df = locate(image, **params)
        if df is not None and not df.empty:
            df['frame'] = i  # Add the frame index manually
            all_dfs.append(df)
            
    return pd.concat(all_dfs, ignore_index=True)