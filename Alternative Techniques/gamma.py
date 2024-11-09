import cv2
import numpy as np
import glob
import os


def adaptive_gamma_correction(image):
    """
    Apply adaptive gamma correction to the image.

    Args:
    image (numpy array): Input image.

    Returns:
    numpy array: Gamma-corrected image.
    """
    if len(image.shape) == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2].astype(np.float32)
    else:
        v_channel = image.astype(np.float32)

    contrast = np.max(v_channel) - np.min(v_channel)

    if contrast <= 50:
        r = np.log1p(v_channel)
    else:
        r = np.power(v_channel / 255.0, 2.0)

    c = 1.0
    processed_image = np.power(r, c) * 255.0
    processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)

    if len(image.shape) == 3:
        hsv[:, :, 2] = processed_image
        processed_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return processed_image


def main(dataset_directory, output_directory):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    image_files = [file for file in glob.glob(dataset_directory + '/*')
                   if os.path.isfile(file) and os.path.splitext(file)[1].lower() in image_extensions]

    for image_file in image_files:
        image = cv2.imread(image_file)
        gamma_corrected = adaptive_gamma_correction(image)
        output_path = os.path.join(
            output_directory, os.path.basename(image_file))
        cv2.imwrite(output_path, gamma_corrected)

    print("Gamma correction completed.")


if __name__ == "__main__":
    dataset_directory = os.sys.argv[1]
    output_directory = os.sys.argv[2]
    main(dataset_directory, output_directory)
