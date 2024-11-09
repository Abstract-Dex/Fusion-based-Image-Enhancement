import cv2
import numpy as np
import os
import glob
import random


def calculate_clip_limit(hist):
    """
    Calculate the clip limit based on the histogram.

    Args:
    hist (numpy array): Normalized histogram of the L channel.

    Returns:
    int: Clip limit value.
    """
    cdf = np.cumsum(hist)
    min_limit = 0.02 * np.sum(hist)
    max_limit = 0.04 * np.sum(hist)
    min_index = np.abs(cdf - min_limit).argmin()
    max_index = np.abs(cdf - max_limit).argmin()
    clip_limit = max_index - min_index
    # Ensure clip limit is within the desired range
    return max(2, min(4, clip_limit))


def apply_clahe(image):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to the image.

    Args:
    image (numpy array): Input image.

    Returns:
    numpy array: CLAHE-enhanced image.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    hist = cv2.calcHist([l_channel], [0], None, [256], [0, 256])
    hist /= np.sum(hist)
    clip_limit = calculate_clip_limit(hist)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    clahe_img = clahe.apply(l_channel)
    CLAHE_img = cv2.merge((clahe_img, lab[:, :, 1], lab[:, :, 2]))
    return cv2.cvtColor(CLAHE_img, cv2.COLOR_LAB2BGR)


def main(dataset_directory, output_directory):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    all_files = glob.glob(dataset_directory + '/*')
    random.seed(42)
    selected_files = random.sample(all_files, 100)

    for image_file in selected_files:
        if image_file.endswith(tuple(image_extensions)):
            image = cv2.imread(image_file)
            clahe_enhanced = apply_clahe(image)
            output_path = os.path.join(
                output_directory, os.path.basename(image_file))
            cv2.imwrite(output_path, clahe_enhanced)

    print("CLAHE enhancement completed.")


if __name__ == "__main__":
    dataset_directory = os.sys.argv[1]
    output_directory = os.sys.argv[2]
    main(dataset_directory, output_directory)
