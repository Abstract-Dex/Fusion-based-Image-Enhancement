import math
import sys
from skimage import io, color, filters
import os
import numpy as np


def nmetrics(rgb_image):
    """
    Calculate UIQM and UCIQE metrics for a given RGB image.

    Parameters:
    rgb_image (numpy array): RGB image data

    Returns:
    uiqm (float): UIQM metric value
    uciqe (float): UCIQE metric value
    """
    lab_image = color.rgb2lab(rgb_image)
    gray_image = color.rgb2gray(rgb_image)

    # UCIQE
    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    l = lab_image[:, :, 0]

    chroma = np.sqrt(lab_image[:, :, 1]**2 + lab_image[:, :, 2]**2)
    uc = np.mean(chroma)
    sc = np.std(chroma)

    top = int(np.round(0.01 * l.shape[0] * l.shape[1]))
    sl = np.sort(l, axis=None)
    isl = sl[::-1]
    conl = np.mean(isl[::top]) - np.mean(sl[::top])

    satur = [chroma[i] / l[i] if chroma[i] !=
             0 and l[i] != 0 else 0 for i in range(len(l))]
    us = np.mean(satur)

    uciqe = c1 * sc + c2 * conl + c3 * us

    # UIQM
    p1, p2, p3 = 0.0282, 0.2953, 3.5753

    # 1st term UICM
    rg = rgb_image[:, :, 0] - rgb_image[:, :, 1]
    yb = (rgb_image[:, :, 0] + rgb_image[:, :, 1]) / 2 - rgb_image[:, :, 2]
    rgl = np.sort(rg, axis=None)
    ybl = np.sort(yb, axis=None)
    al1, al2 = 0.1, 0.1
    T1, T2 = int(al1 * len(rgl)), int(al2 * len(rgl))
    rgl_tr = rgl[T1:-T2]
    ybl_tr = ybl[T1:-T2]

    urg = np.mean(rgl_tr)
    s2rg = np.std(rgl_tr)
    uyb = np.mean(ybl_tr)
    s2yb = np.std(ybl_tr)

    uicm = -0.0268 * np.sqrt(urg**2 + uyb**2) + 0.1586 * np.sqrt(s2rg + s2yb)

    # 2nd term UISM
    Rsobel = rgb_image[:, :, 0] * filters.sobel(rgb_image[:, :, 0])
    Gsobel = rgb_image[:, :, 1] * filters.sobel(rgb_image[:, :, 1])
    Bsobel = rgb_image[:, :, 2] * filters.sobel(rgb_image[:, :, 2])

    Rsobel = np.round(Rsobel).astype(np.uint8)
    Gsobel = np.round(Gsobel).astype(np.uint8)
    Bsobel = np.round(Bsobel).astype(np.uint8)

    Reme = eme(Rsobel)
    Geme = eme(Gsobel)
    Beme = eme(Bsobel)

    uism = 0.299 * Reme + 0.587 * Geme + 0.114 * Beme

    # 3rd term UIConM
    uiconm = logamee(gray_image)

    uiqm = p1 * uicm + p2 * uism + p3 * uiconm
    return uiqm, uciqe


def eme(ch, blocksize=8):
    """
    Calculate the entropy of an image block.

    Parameters:
    ch (numpy array): Image block data
    blocksize (int): Block size (default=8)

    Returns:
    eme (float): Entropy value
    """
    num_x, num_y = math.ceil(
        ch.shape[0] / blocksize), math.ceil(ch.shape[1] / blocksize)
    eme = 0
    w = 2. / (num_x * num_y)

    for i in range(num_x):
        xlb, xrb = i * blocksize, (i+1) * \
            blocksize if i < num_x - 1 else ch.shape[0]
        for j in range(num_y):
            ylb, yrb = j * blocksize, (j+1) * \
                blocksize if j < num_y - 1 else ch.shape[1]
            block = ch[xlb:xrb, ylb:yrb]
            blockmin, blockmax = float(np.min(block)), float(np.max(block))

            if blockmin == 0:
                blockmin += 1
            if blockmax == 0:
                blockmax += 1
            eme += w * math.log(blockmax / blockmin)
    return eme


def plipsum(i, j, gamma=1026):
    return i + j - i * j / gamma


def plipsub(i, j, k=1026):
    return k * (i - j) / (k - j)


def plipmult(c, j, gamma=1026):
    return gamma - gamma * (1 - j / gamma)**c


def logamee(ch, blocksize=8):
    """
    Calculate the logarithmic amplitude of an image block.

    Parameters:
    ch (numpy array): Image block data
    blocksize (int): Block size (default=8)

    Returns:
    logamee (float): Logarithmic amplitude value
    """
    num_x, num_y = math.ceil(
        ch.shape[0] / blocksize), math.ceil(ch.shape[1] / blocksize)
    s = 0
    w = 1. / (num_x * num_y)

    for i in range(num_x):
        xlb, xrb = i * blocksize, (i+1) * \
            blocksize if i < num_x - 1 else ch.shape[0]
        for j in range(num_y):
            ylb, yrb = j * blocksize, (j+1) * \
                blocksize if j < num_y - 1 else ch.shape[1]
            block = ch[xlb:xrb, ylb:yrb]
            blockmin, blockmax = float(np.min(block)), float(np.max(block))

            top = plipsub(blockmax, blockmin)
            bottom = plipsum(blockmax, blockmin)

            if bottom == 0:
                m = 0
            else:
                m = top / bottom
            if m == 0.:
                s += 0
            else:
                s += m * math.log(m)
    return plipmult(w, s)


def Get_Mean_Uiqm_UCIQE(result_path):
    """
    Calculate the mean UIQM and UCIQE metrics for a set of images.

    Parameters:
    result_path (str): Path to the image directory

    Returns:
    muiqm (float): Mean UIQM value
    muciqe (float): Mean UCIQE value
    """
    result_dirs = os.listdir(result_path)[:5]
    sumuiqm, sumuciqe = 0., 0.
    N = 0

    for imgdir in result_dirs:
        if '.jpg' in imgdir:
            corrected = io.imread(os.path.join(result_path, imgdir))
            uiqm, uciqe = nmetrics(corrected)
            sumuiqm += uiqm
            sumuciqe += uciqe
            N += 1

    muiqm = sumuiqm / N
    muciqe = sumuciqe / N
    return muiqm, muciqe


def main(input_path):
    """
    Main function to calculate the mean UIQM and UCIQE metrics for a set of images.

    Parameters:
    input_path (str): Path to the image directory
    output_path (str): Path to the output directory
    """
    muiqm, muciqe = Get_Mean_Uiqm_UCIQE(input_path)
    print("Mean UIQM:", muiqm)
    print("Mean UCIQE:", muciqe)


if __name__ == "__main__":
    input_path = sys.argv[1]
    main(input_path)
