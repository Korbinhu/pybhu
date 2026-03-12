from .viewer import ImageStackViewer


def view_stack(data, **options):
    viewer = ImageStackViewer(data, **options)
    viewer.show()
    return viewer
