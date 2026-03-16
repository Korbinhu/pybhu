from .viewer import ImageStackViewer
from .loader import load_data


def img_viewer(data=None, **options):
    """
    Launch the ImageStackViewer.
    data can be a numpy array, a path to a .npy, .pkl, or .npz file, or None (defaults to zeros).
    """
    if isinstance(data, (str, bytes)):
        data = load_data(data)
    
    viewer = ImageStackViewer(data, **options)
    viewer.show()
    return viewer
