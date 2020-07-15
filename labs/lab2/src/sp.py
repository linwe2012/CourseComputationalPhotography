"""
Original: https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
modified by leon: allowing s&p on multiple channels
"""

import numpy as np
import random
import cv2

def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            def mod(c):
                if len(image.shape) == 3:
                    for ch in range(image.shape[2]):
                        output[i][j][ch] = c
                else:
                    output[i][j] = c
            rdn = random.random()
            if rdn < prob:
                mod(0)
            elif rdn > thres:
                mod(255)
            else:
                output[i][j] = image[i][j]
    return output

image = cv2.imread('OpenCVHW1/Lena.png') # Only for grayscale image
noise_img = sp_noise(image,0.01)
cv2.imwrite('OpenCVHW1/Lena_sp.png', noise_img)
print(image.shape)