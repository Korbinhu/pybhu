from .viewer import ImageStackViewer


def img_viewer(data, **options):
    viewer = ImageStackViewer(data, **options)
    viewer.show()
    return viewer
