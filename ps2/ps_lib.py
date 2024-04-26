from pathlib import Path
from typing import Tuple, Union

import numpy as np
from PIL import Image

__all__ = ['read_image', 'write_image', 'pad', 'resize']

#------------------------------------------------------------------------------

def read_image(path: Union[Path, str]) -> np.ndarray:
    '''
    Read a PNG or JPG image an array of linear RGB radiance values ∈ [0,1].
    '''
    return (np.float32(Image.open(path)) / 255)**2.2


def write_image(path: Union[Path, str], image: np.ndarray) -> None:
    '''
    Write an array of linear RGB radiance values ∈ [0,1] as a PNG or JPG image.
    '''
    Image.fromarray(np.uint8(255 * image.clip(0, 1)**(1/2.2))).save(path)


def pad(image: np.ndarray, u_pad: int, v_pad: int) -> np.ndarray:
    '''
    Pad an image using nearest-neighbor extrapolation.

    Parameters:
        image: A single-channel (2D) or multichannel (3D) image
        u_pad: The number of pixels to add to each side, vertically
        v_pad: The number of pixels to add to each side, horizontally

    Returns:
        An image of size `(image.shape[0] + 2×u_pad, image.shape[1] + 2×v_pad)`
    '''
    # Compute measurements.
    u0 = u_pad # Top edge of the original image
    v0 = v_pad # Left edge of the original image
    u1 = u_pad + image.shape[0] # Bottom edge of the original image
    v1 = v_pad + image.shape[1] # Right edge of the original image
    h_padded = image.shape[0] + 2 * u_pad # Padded image height
    w_padded = image.shape[1] + 2 * v_pad # Padded image width

    # Copy the input image into the center of a larger array.
    padded_image = np.zeros((h_padded, w_padded, *image.shape[2:]), np.float32)
    padded_image[u0:u1, v0:v1] = image

    # Fill in the edges.
    padded_image[:u0, v0:v1] = image[:1, :]
    padded_image[u1:, v0:v1] = image[-1:, :]
    padded_image[u0:u1, :v0] = image[:, :1]
    padded_image[u0:u1, v1:] = image[:, -1:]

    # Fill in the corners.
    padded_image[:u0, :v0] = image[0, 0]
    padded_image[:u0, v1:] = image[0, -1]
    padded_image[u1:, :v0] = image[-1, 0]
    padded_image[u1:, v1:] = image[-1, -1]

    # Return the result.
    return padded_image


def resize(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    '''
    Convert an image to the specified size (height, width) using bicubic
    interpolation.
    '''
    return _resize_channel(image, size) if image.ndim == 2 else np.dstack([
        _resize_channel(chan, size) for chan in image.transpose(2, 0, 1)
    ])


def _resize_channel(chan: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(chan).resize(size[::-1], Image.CUBIC))
