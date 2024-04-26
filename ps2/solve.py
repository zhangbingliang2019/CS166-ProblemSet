import numpy as np
from PIL import Image
from ps_lib import read_image, write_image, resize


def load_given_kernel(path):
    return np.load(path)


def pad_image(image, kernel_height, kernel_width):
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    return np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='edge')


def cross_correlation(image, kernel):
    kernel_height, kernel_width = kernel.shape
    padded_image = pad_image(image, kernel_height, kernel_width)
    height, width = image.shape[:2]
    result = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            for k in range(3):
                region = padded_image[i:i + kernel_height, j:j + kernel_width, k]
                result[i, j, k] = np.sum(region * kernel)

    return result


def build_pyramid(image):
    gaussians = [np.array(image, dtype=np.float32)]
    laplacians = []
    current = image
    kernel = np.load('gaussian-kernel.npy')
    while min(current.shape[:2]) > max(kernel.shape):
        smooth = cross_correlation(current, kernel)
        downsize = (smooth.shape[0] // 2, smooth.shape[1] // 2)
        downsampled = resize(smooth, size=downsize)
        gaussians.append(np.array(downsampled, dtype=np.float32))

        expanded = resize(downsampled, size=smooth.shape[:2])
        laplacian = current - expanded
        laplacians.append(laplacian)

        current = downsampled

    laplacians.append(gaussians[-1])
    return np.array(gaussians[::-1], dtype=object), np.array(laplacians[::-1], dtype=object)


def create_mask(height, width):
    mask = np.zeros(shape=(height, width, 3), dtype=np.float32)
    mask[:, :width // 2, :] = 1.0
    return mask


def blend_pyramids(lap1, lap2, mask_pyramid):
    blended_pyramid = []
    for lap1_level, lap2_level, mask_level in zip(lap1, lap2, mask_pyramid):
        blended_level = mask_level * lap1_level + (1 - mask_level) * lap2_level
        blended_pyramid.append(blended_level)
    return blended_pyramid


def reconstruct_from_lap_pyramid(lap_pyramid):
    image = lap_pyramid[0]
    for level in lap_pyramid[1:]:
        image = resize(image, level.shape[:2]) + level
    return image


def compute_gain_factors(images):
    base_image = images[0]  # Darkest image
    base_image_luminance = compute_luminance(base_image)
    factors = []
    for image in images:
        image_luminance = compute_luminance(image)
        # Avoid zero division and saturated pixels
        valid_pixels = (base_image_luminance > 0.05) & (base_image_luminance < 0.95) & (image_luminance > 0.05) & (
                image_luminance < 0.95)
        factor = np.median(base_image_luminance[valid_pixels] / image_luminance[valid_pixels])
        factors.append(factor)
    return factors


def merge_images(images, factors):
    hdr_image = np.zeros_like(images[0])
    for image, factor in zip(images, factors):
        hdr_image += image * factor
    return hdr_image


def render_hdr_images_by_quantile(hdr_image, path):
    levels = np.linspace(0.5, 1, 5, endpoint=False)
    for i, level in enumerate(levels, 1):
        ld = np.quantile(hdr_image, level)
        # hd = np.quantile(hdr_image, level + 0.2)
        # ldr_image = (hdr_image - ld) / (hd - ld)
        ldr_image = hdr_image / ld
        rendered_image = np.clip(ldr_image, 0, 1)
        write_image(path.format(i), rendered_image)


def gaussian(x, sigma):
    return np.exp(-x ** 2 / (2 * sigma ** 2))


def bilateral_filter(raw_img, sigma_domain, sigma_range):
    filtered_image = np.zeros_like(raw_img)
    radius = 5
    diameter = 11
    img = pad_image(raw_img, diameter, diameter)

    print('Applying Bilateral Filter...')

    # Compute Gaussian spatial weights
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    g_domain = gaussian(np.sqrt(x ** 2 + y ** 2), sigma_domain)
    for i in tqdm.trange(raw_img.shape[0]):
        for j in range(raw_img.shape[1]):
            local_region = img[i:i + 2 * radius + 1, j:j + 2 * radius + 1, :]
            # Compute Gaussian range weights
            color_difference = local_region - img[i + radius, j + radius, :]
            g_range = gaussian(np.linalg.norm(color_difference, axis=2), sigma_range)
            # Compute the weights and apply them
            weights = g_domain * g_range
            filtered_image[i, j, :] = np.sum(weights[..., None] * local_region, axis=(0, 1)) / np.sum(weights)
    return filtered_image


def load_hdr_image(image_path):
    return np.load(image_path)


def compute_luminance(image):
    R, G, B = image[..., 0], image[..., 1], image[..., 2]
    Y = 0.3 * R + 0.6 * G + 0.1 * B
    return Y


def compute_chrominance(image, luminance):
    return image / (luminance[..., None] + 1e-6)


def log_scale(luminance):
    return np.log10(luminance + 0.01)


def tone_mapping(hdr_image_path):
    image = load_hdr_image(hdr_image_path)
    image = (image - image.min()) / (image.max() - image.min())
    luminance = compute_luminance(image)
    chrominance = compute_chrominance(image, luminance)

    log_luminance = log_scale(luminance)
    image_width = image.shape[1]
    sigma_domain = image_width / 50
    sigma_range = 0.4

    base_image = bilateral_filter(log_luminance[:, :, None], sigma_domain, sigma_range)[:, :, 0]
    detail_image = log_luminance - base_image

    R_target = 100
    scale_factor = np.log10(R_target) / (np.max(base_image) - np.min(base_image))
    beta = - np.max(scale_factor * base_image)

    alpha_detail = 3
    L_output = scale_factor * base_image + alpha_detail * detail_image + beta

    final_luminance = 10 ** L_output
    final_image = final_luminance[..., None] * chrominance

    return final_image


def average(*args):
    value = np.zeros_like(args[0])
    for v in args:
        value += v
    return value / len(args)


def demosaicing(img):
    height, width = img.shape
    # four patterns in total
    color_img = np.zeros((height - 2, width - 2, 3))

    # (R G R / G B G / R G R) pattern
    color_img[0::2, 0::2] = np.stack([
        average(img[0:-2:2, 0:-2:2], img[0:-2:2, 2::2], img[2::2, 0:-2:2], img[2::2, 2::2]),
        average(img[0:-2:2, 1:-1:2], img[1:-1:2, 0:-2:2], img[2::2, 1:-1:2], img[1:-1:2, 2::2]),
        img[1:-1:2, 1:-1:2],
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


def decode_npz(path):
    img_raw = np.load(path)['data']
    bgr_color_img = demosaicing(img_raw)
    color_img = np.stack([bgr_color_img[:, :, -1], bgr_color_img[:, :, 1], bgr_color_img[:, :, 0]], axis=2)
    # instead of normalizing by maximal value of image, directly use maximal valu in Bayer data, i.e. 2**10
    color_img = color_img / 2 ** 10
    write_image('test.png', color_img)
    return color_img


def Question1():
    image_path = 'apple.png'
    image = read_image(image_path)

    # Gaussian kernel
    gaussian_kernel = np.load('gaussian-kernel.npy')
    gaussian_result = cross_correlation(image, gaussian_kernel)
    write_image('images/q1-gaussian.png', gaussian_result)

    # Sobel vertical kernel
    sobel_vertical_kernel = np.load('sobel-kernel-vertical.npy')
    sobel_vertical_result = cross_correlation(image, sobel_vertical_kernel)
    write_image('images/q1-sobel-vertical.png', sobel_vertical_result)

    # Sobel horizontal kernel
    sobel_horizontal_kernel = np.load('sobel-kernel-horizontal.npy')
    sobel_horizontal_result = cross_correlation(image, sobel_horizontal_kernel)
    write_image('images/q1-sobel-horizontal.png', sobel_horizontal_result)
    print(f'Question 1: save image to images/q1-*')


def Question2():
    apple = read_image('apple.png')
    orange = read_image('orange.png')
    mask = read_image('mask.png')

    # Build pyramids
    apple_gauss, apple_lap = build_pyramid(apple)
    orange_gauss, orange_lap = build_pyramid(orange)
    mask_gauss, mask_lap = build_pyramid(mask)

    np.save('q2-apple.npy', apple_gauss)
    np.save('q2-orange.npy', orange_gauss)
    np.save('q2-mask.npy', mask_gauss)

    for i, level in enumerate(apple_gauss):
        write_image(f'images/q2-apple-{i + 1}.png', level)
    for i, level in enumerate(orange_gauss):
        write_image(f'images/q2-orange-{i + 1}.png', level)
    for i, level in enumerate(mask_gauss):
        write_image(f'images/q2-mask-{i + 1}.png', level)
    print(f'Question 2: save image to images/q2-*')


def Question3():
    apple = read_image('apple.png')
    orange = read_image('orange.png')
    mask = read_image('mask.png')

    # Build pyramids
    apple_gauss, apple_lap = build_pyramid(apple)
    orange_gauss, orange_lap = build_pyramid(orange)
    mask_gauss, mask_lap = build_pyramid(mask)

    # Blend the pyramids
    blended_pyramid = blend_pyramids(apple_lap, orange_lap, mask_gauss)
    blended_image = reconstruct_from_lap_pyramid(blended_pyramid)
    write_image('images/q3-apple-and-orange.png', blended_image)

    new1 = read_image('images/q3-new-1.png')[:, :, :3]  # remove last alpha channel
    new2 = read_image('images/q3-new-2.png')[:, :, :3]
    mask = create_mask(new1.shape[0], new1.shape[1])

    # Build pyramids
    new1_gauss, new1_lap = build_pyramid(new1)
    new2_gauss, new2_lap = build_pyramid(new2)
    mask_gauss, mask_lap = build_pyramid(mask)

    # Blend the pyramids
    blended_pyramid = blend_pyramids(new1_lap, new2_lap, mask_gauss)
    blended_image = reconstruct_from_lap_pyramid(blended_pyramid)
    write_image('images/q3-new-images.png', blended_image)
    print(f'Question 3: save image to images/q3-new-images.png')


def Question4():
    for k in range(4):
        path = f'q4-shot-{k + 1}'
        image = decode_npz(path + '.npz')
        write_image('images/' + path + '.png', image)
    print(f'Question 4: save image to images/q4-*')


def Question5():
    images = [read_image(f'canyon-shot-{i}.png') for i in range(1, 5)]
    factors = compute_gain_factors(images)
    hdr_image = merge_images(images, factors)

    np.save('q5-canyon-composite.npy', hdr_image)
    print(f'Question 5: save npy to q5-canyon-composite.npy')
    render_hdr_images_by_quantile(hdr_image, 'images/q5-canyon-rendering-{}.png')


    images = [read_image(f'images/q4-shot-{i}.png') for i in range(1, 5)]
    factors = compute_gain_factors(images)
    hdr_image = merge_images(images, factors)

    np.save('q5-captured-composite.npy', hdr_image)
    print(f'Question 5: save npy to q5-captured-composite.npy')
    render_hdr_images_by_quantile(hdr_image, 'images/q5-captured-rendering-{}.png')


def Question6():
    image_path = 'noisy-image.png'
    image = read_image(image_path)

    sigma_domain = 1
    sigma_range = 0.3
    filtered_image = bilateral_filter(image, sigma_domain, sigma_range)

    write_image('images/q6-filtered-image.png', filtered_image)
    print(f'Question 6: save image to images/q6-filtered-image.png')


def Question7():
    canyon_image = tone_mapping('q5-canyon-composite.npy')
    write_image('images/q7-canyon.png', canyon_image.clip(0, 1))
    print(f'Question 7: save image to images/q7-canyon.png')

    captured_image = tone_mapping('q5-captured-composite.npy')
    write_image('images/q7-captured.png', captured_image.clip(0, 1))
    print(f'Question 7: save image to images/q7-captured.png')


if __name__ == '__main__':
    Question1()
    Question2()
    Question3()
    Question4()
    Question5()
    Question6()
    Question7()