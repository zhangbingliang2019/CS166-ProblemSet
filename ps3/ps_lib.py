from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

__all__ = ['read_image', 'write_image']

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
