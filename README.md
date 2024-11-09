# Fusion-based-Image-Enhancement

This is an underwater image enhancement technique that overcomes the challenges of of underwater images by combining two state-of-the-art image enhancement techniques: Relative Global Histogram Stretching (RGHS) and Contrast Limited Adaptive Histogram Equalization (CLAHE). The fusion of these two techniques is done using the Euclidean Norm and then normalizing the resultant image.

The technique is tested on a dataset of underwater images and the results are compared with the original images. The datsets used are the Trash ICRA 2019 dataset and the Brackish dataset.

The images are evaluated using the following metrics:

- Underwater Image Quality Measure (UIQM)
- Underwater Color Image Quality Measure (UCIQE)
- Entropy

The `Fusion-Technique.py` file contains the code for the enhancement technique.

The `Alternate Techniques` folder contains the code for the individual techniques and the `Evaluation` folder contains the code for the evaluation of the images.

## Instructions to run the code

1. Clone the repository
2. Run `pip install -r requirements.txt` in the terminal.
3. Run `python Fusion-Technique.py "path-to-data" "path-to-save-results"` in the terminal.
