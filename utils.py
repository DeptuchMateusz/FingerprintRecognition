import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import cv2 as cv
import math
import scipy.ndimage
import pandas as pd

def normalize_image(input_img, target_mean = 100, target_var = 100):
    img_mean = np.mean(input_img)
    img_var = np.var(input_img)
    height, width = input_img.shape
    result = input_img.copy()

    for row in range(height):
        for col in range(width):
            pixel = input_img[row, col]
            diff = pixel - img_mean
            scale = sqrt((target_var * (diff ** 2)) / img_var)
            result[row, col] = target_mean + scale if diff > 0 else target_mean - scale

    return result

def segment_and_normalize(image, block_size=9, std_ratio=0.2):
    height, width = image.shape
    roi_threshold = np.std(image) * std_ratio

    local_std_map = np.zeros_like(image, dtype=np.float32)
    roi_mask = np.ones_like(image, dtype=np.uint8)
    masked_img = image.copy()

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            y_end = min(y + block_size, height)
            x_end = min(x + block_size, width)
            block = image[y:y_end, x:x_end]
            std_block = np.std(block)
            local_std_map[y:y_end, x:x_end] = std_block

    roi_mask[local_std_map < roi_threshold] = 0

    ellipse_center_y = (50 + 320) // 2
    ellipse_center_x = (50 + 220) // 2
    ellipse_height = 320 - 50
    ellipse_width = 220 - 50
    
    y_coords, x_coords = np.ogrid[:height, :width]
    ellipse_mask = ((x_coords - ellipse_center_x) ** 2 / (ellipse_width/2) ** 2 + 
                    (y_coords - ellipse_center_y) ** 2 / (ellipse_height/2) ** 2) <= 1
    
    roi_mask[ellipse_mask] = 1

    morph_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (block_size * 2, block_size * 2))
    roi_mask = cv.morphologyEx(roi_mask, cv.MORPH_OPEN, morph_kernel)
    roi_mask = cv.morphologyEx(roi_mask, cv.MORPH_CLOSE, morph_kernel)
    
    roi_mask[ellipse_mask] = 1

    masked_img *= roi_mask

    background_pixels = image[roi_mask == 0]
    if len(background_pixels) > 0:
        bg_mean = np.mean(background_pixels)
        bg_std = np.std(background_pixels)
        if bg_std > 0:
            normalized_img = (image - bg_mean) / bg_std
        else:
            normalized_img = image - bg_mean
    else:
        normalized_img = (image - np.mean(image)) / np.std(image)

    return masked_img, normalized_img, roi_mask

def estimate_orientation_map(image, block_size=16, apply_smoothing=False):

    calc_numerator = lambda gx, gy: 2 * gx * gy
    calc_denominator = lambda gx, gy: gx ** 2 - gy ** 2

    height, width = image.shape

    sobel_y = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=int)
    sobel_x = sobel_y.T

    # Compute gradients (normalized to avoid overflows)
    gx_full = cv.filter2D(image / 125.0, -1, sobel_y) * 125
    gy_full = cv.filter2D(image / 125.0, -1, sobel_x) * 125

    orientation_map = []

    for row in range(1, height, block_size):
        orientation_row = []
        for col in range(1, width, block_size):
            num = 0
            den = 0

            for y in range(row, min(row + block_size, height - 1)):
                for x in range(col, min(col + block_size, width - 1)):
                    gx = round(gx_full[y, x])
                    gy = round(gy_full[y, x])
                    num += calc_numerator(gx, gy)
                    den += calc_denominator(gx, gy)

            if num != 0 or den != 0:
                angle = (math.pi + math.atan2(num, den)) / 2
            else:
                angle = 0
            orientation_row.append(angle)
        orientation_map.append(orientation_row)

    angle_array = np.array(orientation_map)

    if apply_smoothing:
        pass

    return angle_array

def render_orientation_lines(image, roi_mask, angle_grid, block_size=16):

    height, width = image.shape
    output_img = np.zeros_like(image, dtype=np.uint8)
    output_img = cv.cvtColor(output_img, cv.COLOR_GRAY2RGB)

    mask_threshold = (block_size - 1) ** 2

    for x in range(1, width, block_size):
        for y in range(1, height, block_size):
            mask_block = roi_mask[y - 1:y + block_size, x - 1:x + block_size]
            if np.sum(mask_block) > mask_threshold:
                angle = angle_grid[(y - 1) // block_size][(x - 1) // block_size]
                tangent = math.tan(angle)

                if -1 <= tangent <= 1:
                    start = (x, int((-block_size / 2) * tangent + y + block_size / 2))
                    end = (x + block_size, int((block_size / 2) * tangent + y + block_size / 2))
                else:
                    start = (int(x + block_size / 2 + block_size / (2 * tangent)), y + block_size // 2)
                    end = (int(x + block_size / 2 - block_size / (2 * tangent)), y - block_size // 2)

                cv.line(output_img, start, end, color=(150, 150, 150), thickness=1)

    return output_img

def ridge_frequency(image, mask, orientation_map, block_size=16, kernel_size=5, min_wavelength=5, max_wavelength=5):

    height, width = image.shape
    freq_image = np.zeros_like(image, dtype=np.float32)

    for y in range(0, height - block_size, block_size):
        for x in range(0, width - block_size, block_size):
            block = image[y:y + block_size, x:x + block_size]
            orientation = orientation_map[y // block_size][x // block_size]

            # Compute average orientation in block
            cos2theta = np.cos(2 * orientation)
            sin2theta = np.sin(2 * orientation)
            avg_orientation = math.atan2(sin2theta, cos2theta) / 2

            # Rotate block to make ridges vertical
            rotated = scipy.ndimage.rotate(block, angle=avg_orientation * 180 / np.pi + 90,
                                           reshape=False, order=3, mode='nearest')

            # Crop center to avoid rotation artifacts
            crop_size = int(np.fix(block_size / np.sqrt(2)))
            offset = (block_size - crop_size) // 2
            cropped = rotated[offset:offset + crop_size, offset:offset + crop_size]

            # Project along columns
            ridge_proj = np.sum(cropped, axis=0)
            dilated = scipy.ndimage.grey_dilation(ridge_proj, kernel_size, structure=np.ones(kernel_size))
            ridge_noise = np.abs(dilated - ridge_proj)

            # Peak detection
            peaks = (ridge_noise < 2) & (ridge_proj > np.mean(ridge_proj))
            peak_indices = np.where(peaks)[0]

            if len(peak_indices) >= 2:
                wave_len = (peak_indices[-1] - peak_indices[0]) / (len(peak_indices) - 1)
                if min_wavelength <= wave_len <= max_wavelength:
                    freq_val = 1.0 / wave_len
                else:
                    freq_val = 0
            else:
                freq_val = 0

            freq_image[y:y + block_size, x:x + block_size] = freq_val

    # Apply mask and compute median of valid frequency values
    freq_image *= mask
    valid_freqs = freq_image[freq_image > 0]
    median_frequency = np.median(valid_freqs) if valid_freqs.size else 0

    return median_frequency * mask

def gabor_filter(image, orientation, frequency, kx=0.65, ky=0.65):

    angle_step = 3  
    image = np.double(image)
    rows, cols = image.shape
    filtered_image = np.zeros_like(image)

    freq_flat = np.round(frequency[frequency > 0] * 100) / 100
    unique_freqs = np.unique(freq_flat)

    gabor_kernels = {}

    for freq_val in unique_freqs:
        sigma_x = kx / freq_val
        sigma_y = ky / freq_val
        size = int(np.round(3 * max(sigma_x, sigma_y)))
        x, y = np.meshgrid(np.arange(-size, size+1), np.arange(-size, size+1))
        base_filter = np.exp(-((x**2 / sigma_x**2) + (y**2 / sigma_y**2))) * np.cos(2 * np.pi * freq_val * x)

        for angle_deg in range(0, 180, angle_step):
            rot_filter = scipy.ndimage.rotate(base_filter, -(angle_deg + 90), reshape=False)
            gabor_kernels[(freq_val, angle_deg)] = rot_filter

    block_size = 16
    orient_idx = np.round(orientation * 180 / np.pi / angle_step).astype(int)
    orient_idx %= (180 // angle_step)  # wrap around

    pad = max([kernel.shape[0] // 2 for kernel in gabor_kernels.values()]) 
    for i in range(pad, rows - pad):
        for j in range(pad, cols - pad):
            freq_val = np.round(frequency[i, j] * 100) / 100
            if freq_val > 0:
                angle = orient_idx[i // block_size, j // block_size] * angle_step
                kernel = gabor_kernels.get((freq_val, angle))
                if kernel is not None:
                    ksize = kernel.shape[0] // 2
                    patch = image[i - ksize:i + ksize + 1, j - ksize:j + ksize + 1]
                    if patch.shape == kernel.shape:
                        filtered_image[i, j] = np.sum(patch * kernel)

    # Binarize or enhance filtered result
    result_img = 255 - (filtered_image < 0).astype(np.uint8) * 255

    return result_img

def k3m_skeleton(img, mask=None):

    def neighbours(x, y, image):
        return [image[x-1, y], image[x-1, y+1], image[x, y+1], image[x+1, y+1],
            image[x+1, y], image[x+1, y-1], image[x, y-1], image[x-1, y-1]]

    def transitions(neighbors):
        n = neighbors + neighbors[0:1]
        return sum((n[i] == 0 and n[i+1] == 1) for i in range(8))
    
    img = (img > 0).astype(np.uint8)
    img = np.pad(img, ((1,1),(1,1)), 'constant', constant_values=0)
    
    changed = True
    while changed:
        changed = False
        for pattern in [
            [0, 2, 4, 6],
            [0], [2], [4], [6],
        ]:
            to_remove = []
            for x in range(1, img.shape[0] - 1):
                for y in range(1, img.shape[1] - 1):
                    if img[x, y] == 1:
                        nb = neighbours(x, y, img)
                        if 2 <= sum(nb) <= 6 and transitions(nb) == 1:
                            if any(nb[i] == 0 for i in pattern):
                                to_remove.append((x, y))
            if to_remove:
                changed = True
                for x, y in to_remove:
                    img[x, y] = 0

    skel = (img[1:-1, 1:-1] * 255).astype(np.uint8)
    skel = cv.bitwise_not(skel)

    if mask is not None:
        mask_bin = (mask > 0).astype(np.uint8) * 255
        skel = cv.bitwise_and(skel, mask_bin)
        skel[mask_bin == 0] = 255
    return skel

def morphological_skeleton(image):
    img = image.copy()

    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    _, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    
    kernel_noise = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel_noise, iterations=1)

    skel = np.zeros(img.shape, np.uint8)
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))

    while True:
        open_img = cv.morphologyEx(img, cv.MORPH_OPEN, element)
        temp = cv.subtract(img, open_img)
        eroded = cv.erode(img, element)
        skel = cv.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv.countNonZero(img) == 0:
            break

    skel = cv.bitwise_not(skel)
    
    kernel_vertical = np.array([[1, 0],
                                [1, 0]], dtype=np.uint8)
    
    kernel_horizontal = np.array([[1, 1],
                                  [0, 0]], dtype=np.uint8)
    
    kernel_diag1 = np.array([[1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]], dtype=np.uint8)
    
    kernel_diag2 = np.array([[0, 0, 1],
                             [0, 1, 0],
                             [1, 0, 0]], dtype=np.uint8)
    
    for kernel in [kernel_vertical, kernel_horizontal, kernel_diag1, kernel_diag2]:
        skel = cv.morphologyEx(skel, cv.MORPH_OPEN, kernel, iterations=1)

    skel[:7, :] = 255
    skel[-7:, :] = 255
    skel[:, :7] = 255
    skel[:, -7:] = 255 

    return skel

# def extract_minutiae(image, kernel_size=3, threshold=10):

#     binary = (image < threshold).astype(np.uint8)

#     output = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
#     height, width = image.shape
#     color_map = {"ending": (220, 0, 0), "bifurcation": (0, 220, 0)}

#     if kernel_size == 3:
#         offsets = [(-1, -1), (-1, 0), (-1, 1),
#                    ( 0, 1), (1, 1), (1, 0),
#                    (1, -1), (0, -1), (-1, -1)]
#     else:
#         offsets = [(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
#                    (-1, 2), (0, 2), (1, 2), (2, 2),
#                    (2, 1), (2, 0), (2, -1), (2, -2),
#                    (1, -2), (0, -2), (-1, -2), (-2, -2)]

#     for y in range(kernel_size, height - kernel_size):
#         for x in range(kernel_size, width - kernel_size):
#             if binary[y, x] == 1:
#                 neighbors = [binary[y + dy, x + dx] for dx, dy in offsets]
#                 transitions = sum((neighbors[i] != neighbors[i + 1]) for i in range(len(neighbors) - 1)) // 2

#                 if transitions == 1:
#                     cv.circle(output, (x, y), 2, color_map["ending"], -1)  
#                 elif transitions == 3:
#                     cv.circle(output, (x, y), 2, color_map["bifurcation"], -1)  

#     return output


def extract_minutiae(image, kernel_size=3, threshold=10, draw_arrows=False):
    binary = (image < threshold).astype(np.uint8)
    output = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    height, width = image.shape
    color_map = {"ending": (220, 0, 0), "bifurcation": (0, 220, 0)}
    minutiae_data = []

    if kernel_size == 3:
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   (0, 1), (1, 1), (1, 0),
                   (1, -1), (0, -1), (-1, -1)]
    else:
        offsets = [(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
                   (-1, 2), (0, 2), (1, 2), (2, 2),
                   (2, 1), (2, 0), (2, -1), (2, -2),
                   (1, -2), (0, -2), (-1, -2), (-2, -2)]

    for y in range(kernel_size, height - kernel_size):
        for x in range(kernel_size, width - kernel_size):
            if binary[y, x] == 1:
                neighbors = [binary[y + dy, x + dx] for dx, dy in offsets]
                transitions = sum((neighbors[i] != neighbors[i + 1]) for i in range(len(neighbors) - 1)) // 2

                if transitions == 1:
                    angle = None
                    if draw_arrows:
                        for dx, dy in offsets:
                            if binary[y + dy, x + dx] == 1:
                                angle = math.atan2(dy, dx)
                                x2 = int(x + 15 * math.cos(angle))
                                y2 = int(y + 15 * math.sin(angle))
                                cv.arrowedLine(output, (x, y), (x2, y2), color_map["ending"], 1, tipLength=0.3)
                                break
                    cv.circle(output, (x, y), 3, color_map["ending"], -1)
                    minutiae_data.append({"x": x, "y": y, "type": "ending", "angle": angle})

                elif transitions == 3:
                    angle = None
                    if draw_arrows:
                        dirs = [(dx, dy) for dx, dy in offsets if binary[y + dy, x + dx] == 1]
                        if dirs:
                            dx_avg = -np.mean([dx for dx, dy in dirs])
                            dy_avg = -np.mean([dy for dx, dy in dirs])
                            angle = math.atan2(dy_avg, dx_avg)
                            x2 = int(x + 15 * math.cos(angle))
                            y2 = int(y + 15 * math.sin(angle))
                            cv.arrowedLine(output, (x, y), (x2, y2), color_map["bifurcation"], 1, tipLength=0.3)
                    cv.circle(output, (x, y), 3, color_map["bifurcation"], -1)
                    minutiae_data.append({"x": x, "y": y, "type": "bifurcation", "angle": angle})

    df = pd.DataFrame(minutiae_data)
    return output, df

