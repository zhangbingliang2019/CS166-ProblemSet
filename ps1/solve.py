from pathlib import Path
import numpy as np
from PIL import Image

def decode(byte_data):
    '''
        an array of bytes -> gray img
    '''
    # decode height and width
    height = decode_int(byte_data[0:4])
    width = decode_int(byte_data[4:8])

    # print('The image is in shape ({}, {})'.format(height, width))

    # decode main data
    img = np.zeros((height, width))
    for idx, p in enumerate(range(8, len(byte_data), 2)):
        img[idx//width, idx%width] = decode_int(byte_data[p:p+2])
    return img

def decode_int(byte_list):
    value, base = 0, 256
    for byte in byte_list[::-1]:
        value = value * base + byte
    return value

def average(*args):
    value = np.zeros_like(args[0])
    for v in args:
        value += v
    return value / len(args)

def demosaicing(img):
    height, width = img.shape
    # four patterns in total
    color_img = np.zeros((height-2, width-2, 3))

    # (R G R / G B G / R G R) pattern
    color_img[0::2, 0::2] = np.stack([
        average(img[0:-2:2, 0:-2:2], img[0:-2:2, 2::2], img[2::2, 0:-2:2], img[2::2, 2::2]),
        average(img[0:-2:2, 1:-1:2], img[1:-1:2, 0:-2:2], img[2::2, 1:-1:2], img[1:-1:2, 2::2]),
        img[1:-1:2,1:-1:2], 
    ], axis=-1)
    
    # (G R G / B G B / G R G) pattern
    color_img[0::2, 1::2] = np.stack([
        average(img[0:-2:2, 2::2], img[2::2, 2::2]),
        img[1:-1:2, 2::2],
        average(img[1:-1:2, 1:-1:2], img[1:-1:2, 3::2]),
    ], axis=-1)

    # (G B G / R G R / G B G) pattern
    color_img[1::2, 0::2] = np.stack([
        average(img[2::2, 0:-2:2], img[2::2, 2::2]),
        img[2::2, 1:-1:2],
        average(img[1:-1:2, 1:-1:2], img[3::2, 1:-1:2]),
    ], axis=-1) 

    # (B G B / G R G / B G B) pattern
    color_img[1::2, 1::2] = np.stack([
        img[2::2, 2::2],
        average(img[1:-1:2, 2::2], img[3::2, 2::2], img[2::2, 1:-1:2], img[2::2, 3::2]),
        average(img[1:-1:2, 1:-1:2], img[3::2, 1:-1:2], img[1:-1:2, 3::2], img[3::2, 3::2]),
    ], axis=-1)

    return color_img

def normalizing(img):
    img_min = img.min(keepdims=True)
    img_max = img.max(keepdims=True)
    return (img - img_min) / (img_max - img_min)

def gamma_encoding(img, gamma=1/2.2):
    return img ** gamma

def gamma_decoding(img, gamma=1/2.2):
    return img ** (1 / gamma)

def rgb_to_ycbcr(rgb_img):
    y = rgb_img[:,:,0] * 0.3 + rgb_img[:,:,1] * 0.6 + rgb_img[:,:,2] * 0.1
    cb = (rgb_img[:,:,2] - y) / 1.8
    cr = (rgb_img[:,:,0] - y) / 1.4
    return np.stack([y, cb, cr], axis=2)

def ycbcr_to_rgb(ycbcr_img):
    b = 1.8 * ycbcr_img[:, :, 1] + ycbcr_img[:, :, 0]
    r = 1.4 * ycbcr_img[:, :, 2] + ycbcr_img[:, :, 0]
    g = (ycbcr_img[:, :, 0] - 0.3 * r - 0.1 * b) / 0.6
    return np.stack([r, g, b], axis=2)

def load_img(path):
    return np.float32(Image.open(path)) / 255

def save_img(path, img):
    if len(img.shape)==2:
        Image.fromarray(np.uint8(255 * img.clip(0, 1)), mode='L').save(path)
    else:
        Image.fromarray(np.uint8(255 * img.clip(0, 1))).save(path)


def Question3():
    raw = Path('sample-image.raw').read_bytes()
    byte_data = np.frombuffer(raw, dtype=np.uint8)
    img = decode(byte_data)
    img = normalizing(img)
    path = 'images/q3.png'
    save_img(path, img)
    print(f'Question 3: save image to {path}')
    

def Question4():
    raw = Path('sample-image.raw').read_bytes()
    byte_data = np.frombuffer(raw, dtype=np.uint8)
    img = decode(byte_data)
    
    color_img = demosaicing(img)
    color_img = gamma_encoding(normalizing(color_img))
    path = 'images/q4.png'
    save_img(path, color_img)
    print(f'Question 4: save image to {path}')


def Question5():
    rgb_img = gamma_decoding(load_img('images/q5-original.png'))
    ycbcr_img = rgb_to_ycbcr(rgb_img)

    gray_ycbcr_img = ycbcr_img * np.array([1, 0, 0]).reshape((1, 1, 3))
    gray_img = gamma_encoding(ycbcr_to_rgb(gray_ycbcr_img))
    path = 'images/q5-gray.png'
    save_img(path, gray_img)
    print(f'Question 5: save image to {path}')

    adaptation_ycbcr_img = np.stack([np.ones_like(ycbcr_img[:,:,0])*0.5, -ycbcr_img[:,:,1], -ycbcr_img[:,:,2]], axis=2)
    adaptation_img = gamma_encoding(ycbcr_to_rgb(adaptation_ycbcr_img).clip(0, 1))
    path = 'images/q5-adaptation.png'
    save_img(path, adaptation_img)
    print(f'Question 5: save image to {path}')


def Question6():
    img_raw = np.load('captured-image.npz')['data']
    # BGGR filter
    bgr_color_img = demosaicing(img_raw)
    color_img = np.stack([bgr_color_img[:,:,-1], bgr_color_img[:,:,1], bgr_color_img[:, :, 0]], axis=2)
    color_img = gamma_encoding(normalizing(color_img))
    path = 'images/q6.png'
    save_img(path, color_img)
    print(f'Question 6: save image to {path}')

def Question6():
    img_raw = np.load('q6.npz')['data']
    # BGGR filter
    bgr_color_img = demosaicing(img_raw)
    color_img = np.stack([bgr_color_img[:,:,-1], bgr_color_img[:,:,1], bgr_color_img[:, :, 0]], axis=2)
    # instead of normalizing by maximal value of image, directly use maximal valu in Bayer data, i.e. 2**10
    color_img = gamma_encoding(color_img / 2**10)
    path = 'images/q6.png'
    save_img(path, color_img)
    print(f'Question 6: save image to {path}')

if __name__ == '__main__':
    Question3()
    Question4()
    Question5()
    Question6()
    