import os
import cv2
import numpy as np


def apply_relative_global_histogram_stretching(image):
    """Apply Relative Global Histogram Stretching to the image"""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


def apply_clahe(image):
    """Apply Contrast Limited Adaptive Histogram Equalization to the image"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)


def combine_with_euclidean_norm(image1, image2):
    """Combine two images using Euclidean norm"""
    image1, image2 = image1.astype(np.float32), image2.astype(np.float32)
    combined_image = np.sqrt(np.square(image1) + np.square(image2))
    return ((combined_image - combined_image.min()) /
            (combined_image.max() - combined_image.min()) * 255).astype(np.uint8)


def normalize_image(image):
    """Normalize the image to the 0-255 range"""
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)


def main(input_folder, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_filename in os.listdir(input_folder):
        if image_filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(input_folder, image_filename)
            img = cv2.imread(image_path)

            img1 = apply_relative_global_histogram_stretching(img)
            img2 = apply_clahe(img)
            img3 = combine_with_euclidean_norm(img1, img2)
            img4 = normalize_image(img3)

            output_image_path = os.path.join(output_folder, image_filename)
            cv2.imwrite(output_image_path, img4)

    print("Enhancement completed.")


if __name__ == "__main__":
    input_folder = os.sys.argv[1]
    output_folder = os.sys.argv[2]
    main(input_folder, output_folder)
