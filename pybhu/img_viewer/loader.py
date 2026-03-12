import numpy as np
import pickle
import os

def load_data(path: str) -> np.ndarray:
    """
    Load data from a .npy, .pkl, or .npz file.
    Returns a numpy.ndarray.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.npy':
        data = np.load(path)
        return to_numpy(data)
    elif ext in ['.pkl', '.npz']:
        obj = get_archive_contents(path)
        showable = find_showable_data(obj)
        if showable:
            return to_numpy(showable[0][1])
        raise ValueError(f"No showable data found in {ext} file.")
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Supported: .npy, .pkl, .npz")

def to_numpy(data) -> np.ndarray:
    """Convert input data to a showable numpy array."""
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    if data.ndim not in [2, 3]:
        raise ValueError(f"Data must be 2D or 3D, got {data.ndim}D")
    
    return data

def get_archive_contents(path: str):
    """Load archive and return the raw object or dict."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.pkl':
        with open(path, 'rb') as f:
            return pickle.load(f)
    elif ext == '.npz':
        with np.load(path, allow_pickle=True) as data:
            return dict(data)
    raise ValueError(f"Unsupported archive extension: {ext}")

def find_showable_data(obj, path_prefix=""):
    """
    Recursively find keys/indices in an object that contain showable data.
    Returns a list of tuples (display_name, actual_data).
    """
    showable = []
    
    if isinstance(obj, np.ndarray):
        if obj.ndim in [2, 3]:
            showable.append((path_prefix or "Root Array", obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{path_prefix}/{k}" if path_prefix else str(k)
            showable.extend(find_showable_data(v, new_prefix))
    elif hasattr(obj, 'keys') and hasattr(obj, '__getitem__'): 
        # Support for NpzFile or similar dict-like
        for k in obj.keys():
            v = obj[k]
            new_prefix = f"{path_prefix}/{k}" if path_prefix else str(k)
            showable.extend(find_showable_data(v, new_prefix))
    elif isinstance(obj, (list, tuple)):
        # Only peek at the first element if it's a large list, 
        # or try to convert the whole thing if it's small.
        try:
            arr = np.array(obj)
            if arr.ndim in [2, 3]:
                showable.append((path_prefix or "List Data", arr))
        except Exception:
            # If it's a list of objects, recurse
            if len(obj) < 20: # Limit recursion for safety
                for i, v in enumerate(obj):
                    new_prefix = f"{path_prefix}[{i}]" if path_prefix else f"[{i}]"
                    showable.extend(find_showable_data(v, new_prefix))
                    
    return showable
