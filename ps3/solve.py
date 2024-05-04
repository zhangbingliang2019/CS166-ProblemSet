from PIL import Image
import numpy as np
from ps_lib import read_image, write_image
from scipy.ndimage import shift, gaussian_filter
import tqdm


def to_fivedim_array(light_field_img, macro_pixel_size):
    light_field_img = light_field_img.reshape((
        light_field_img.shape[0] // macro_pixel_size,
        macro_pixel_size,
        light_field_img.shape[1] // macro_pixel_size,
        macro_pixel_size,
        light_field_img.shape[2]
    ))

    light_field = light_field_img.transpose((1, 3, 0, 2, 4))
    return light_field


def get_slice(u, v, light_field, shift_val=7):
    return light_field[u + shift_val, v + shift_val, :, :, :]


def to_twodim_mosaic(light_field):
    mosaic = light_field.transpose((0, 2, 1, 3, 4))
    mosaic = mosaic.reshape((
        light_field.shape[0] * light_field.shape[2],
        light_field.shape[1] * light_field.shape[3],
        light_field.shape[4]
    ))
    return mosaic


def refocus(light_field, depth, shift_val=7):
    final_image = np.zeros_like(light_field[0, 0, :, :, :], dtype=np.float64)
    for u in tqdm.trange(light_field.shape[0]):
        for v in range(light_field.shape[1]):
            slice_uv = light_field[u, v, :, :, :]
            translation = (int(depth * (u - shift_val) + 0.5), -int(depth * (v - shift_val) + 0.5), 0)
            print(translation, end=' ')
            shifted_slice_uv = shift(slice_uv, translation, mode='constant', cval=0)
            final_image += shifted_slice_uv
    return final_image / np.prod(light_field.shape[:2])


def average(light_field):
    return np.mean(light_field, axis=(0, 1))


def compute_high_frequency(image):
    blurred_image = gaussian_filter(image, sigma=2, mode='nearest')
    high_frequency = image - blurred_image
    return high_frequency


def compute_sharpness_weights(high_frequency):
    high_frequency_squared = high_frequency ** 2
    sharpness_weights = gaussian_filter(high_frequency_squared, sigma=8, mode='nearest')
    return sharpness_weights


def compute_all_in_focus_image(images, sharpness_weights):
    all_in_focus_image = np.zeros_like(images[0])
    total_weight = np.zeros_like(all_in_focus_image)

    for image, sharpness_weight in zip(images, sharpness_weights):
        all_in_focus_image += sharpness_weight[..., None] * image
        total_weight += sharpness_weight[..., None]

    all_in_focus_image /= total_weight
    return all_in_focus_image


def to_gray(image):
    return 0.3 * image[:, :, 0] + 0.6 * image[:, :, 1] + 0.1 * image[:, :, 2]


def compute_depth_map(depth_values, sharpness_weights):
    depth_map = np.zeros_like(sharpness_weights[0], dtype=np.float64)
    total_weight = np.zeros_like(depth_map, dtype=np.float64)

    for d in range(len(depth_values)):
        depth_map += sharpness_weights[d] * depth_values[d]
        total_weight += sharpness_weights[d]

    depth_map /= total_weight
    return depth_map


def normalized_cross_correlation(image, kernel):
    kernel_mean = np.mean(kernel)
    kernel_std = np.std(kernel)
    normed_kernel = (kernel - kernel_mean) / (kernel_std + 1e-6)

    image_height, image_width = image.shape[:2]
    kernel_height, kernel_width = kernel.shape[:2]

    ncc_map = np.zeros((image_height - kernel_height, image_width - kernel_width))
    for y in tqdm.trange(image_height - kernel_height):
        for x in range(image_width - kernel_width):
            roi = image[y:y + kernel_height, x:x + kernel_width]
            roi_mean = np.mean(roi)
            roi_std = np.std(roi)
            normed_roi = (roi - roi_mean) / (roi_std + 1e-6)
            ncc_value = np.sum(normed_kernel * normed_roi)
            ncc_map[y, x] = ncc_value
    return ncc_map


def apply_delta_drift(frames, ncc_maps):
    drifts = []
    for ncc_map in ncc_maps:
        y, x = np.unravel_index(ncc_map.argmax(), ncc_map.shape)
        drifts.append((y, x))

    final_image = frames[0]

    for i in tqdm.trange(1, len(frames)):
        translation = (drifts[0][0] - drifts[i][0], drifts[0][1] - drifts[i][1], 0)
        shifted_frame = shift(frames[i], translation, mode='constant', cval=0)
        final_image += shifted_frame
    return final_image / len(frames)


def Question1():
    light_field_img = read_image('light-field.png')
    light_field = to_fivedim_array(light_field_img, 16)

    slice_a = get_slice(-5, -2, light_field)
    slice_b = get_slice(-1, 4, light_field)
    np.save('q1-light-field-slice-a.npy', slice_a)
    np.save('q1-light-field-slice-b.npy', slice_b)

    mosaic = to_twodim_mosaic(light_field)
    write_image('q1-sub-aperture-views.png', mosaic)
    print(f'Question 1: save image to q1-sub-aperture-views.png')


def Question2():
    light_field_img = read_image('light-field.png')
    light_field = to_fivedim_array(light_field_img, 16)

    for i, depth in enumerate(np.linspace(-2, 1, 11)):
        refocus_image = refocus(light_field, depth)
        write_image('q2-depth-{:02d}.png'.format(i), refocus_image)
    print(f'Question 2: save image to q2-depth-*.png')


def Question3():
    image_stack = [read_image('q2-depth-{:02d}.png'.format(i)) for i in range(11)]
    gray_stack = [to_gray(image) for image in image_stack]

    high_frequency_images = [compute_high_frequency(image) for image in gray_stack]
    sharpness_weights = [compute_sharpness_weights(hf_image) for hf_image in high_frequency_images]

    all_in_focus_image = compute_all_in_focus_image(image_stack, sharpness_weights)

    write_image('q3.png', all_in_focus_image)
    print(f'Question 3: save image to q3')


def Question4():
    depth_stack = np.linspace(-2, 1, 11)
    image_stack = [read_image('q2-depth-{:02d}.png'.format(i)) for i in range(11)]
    gray_stack = [to_gray(image) for image in image_stack]

    high_frequency_images = [compute_high_frequency(image) for image in gray_stack]
    sharpness_weights = [compute_sharpness_weights(hf_image) for hf_image in high_frequency_images]

    depth_map = compute_depth_map(depth_stack, sharpness_weights)
    depth_map = depth_map - depth_map.min()
    depth_map = depth_map / depth_map.max()

    write_image('q4.png', depth_map)
    print(f'Question 4: save image to q4')


def Question5():
    frames = [np.load(f'captured-video/{i:06d}.npz')['data'] for i in range(95)]
    frames = np.stack(frames).astype(np.float32) / 255
    kernel = frames[0][160:350, 245:340]
    ncc_maps = [normalized_cross_correlation(frame, kernel) for frame in frames]
    final_image = apply_delta_drift(frames, ncc_maps)
    write_image('q5-composite-image.png', final_image / final_image.max())
    print(f'Question 5: save image to q5-composite-image.png')

if __name__ == '__main__':
    Question1()
    Question2()
    Question3()
    Question4()
    Question5()
