import numpy as np
import pywt


def wavelet_transform(image):
    coeffs2 = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs2
    return np.concatenate((LL.flatten(), LH.flatten(), HL.flatten(), HH.flatten()))


def wavelet_shrinkage(image, threshold=0.1):
    return pywt.threshold(image, threshold, mode='soft')
