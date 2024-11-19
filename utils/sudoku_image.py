import numpy as np
from PIL import Image

def create_sudoku_image(patches):
    """
    Combine 16 MNIST digit patches (4x4 grid) into a single composite image.
    Args:
        patches (list of numpy arrays): 16 MNIST digit images (28x28 each).
    Returns:
        composite_image (numpy array): 112x112 composite image.
    """
    rows = []
    for i in range(4):
        row = np.hstack(patches[i * 4:(i + 1) * 4])  # Combine 4 patches in a row
        rows.append(row)
    composite_image = np.vstack(rows)  # Combine 4 rows into a grid
    return composite_image
