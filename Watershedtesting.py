import PIL.ImageOps
import numpy as np
import cv2
import glob
import os
import scipy.misc
from PIL import Image
import matplotlib.pyplot as plt
from scipy.misc import imsave
from PIL import Image, ImageDraw

import cv2
import sys
import numpy as np
import copy
from PIL import Image
import matplotlib.pyplot as plt
# from PredictFromModel import predict
import operator


def initialBoxes(im):
    '''input: image; return: None'''

    im[im >= 127] = 255
    im[im < 127] = 0

    '''
    # set the morphology kernel size, the number in tuple is the bold pixel size
    kernel = np.ones((2,2),np.uint8)
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    '''

    imgrey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgrey, 127, 255, 0)
    im2, contours, hierachy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)  # RETR_EXTERNAL for only bounding outer box
    # bounding rectangle outside the individual element in image
    res = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # exclude the whole size image and noisy point
        if x is 0: continue
        if w * h < 25: continue

        res.append([(x, y), (x + w, y + h)])
        crop_MainImage = im[y:y + h, x:x + w]
        new_image = Image.fromarray(crop_MainImage)
        new_image = new_image.convert("L")
        plt.imshow(new_image)
        plt.show()
    return res

def main():
    testImagePath = 'C:\\Users\\sabik\\PycharmProjects\\DeepLearningBasics\\TestIMages\\'
    savetestImagePath = 'C:\\Users\\sabik\\PycharmProjects\\DeepLearningBasics\\TestIMages\\'
    testImageFileName = testImagePath + 'Equ1.jpg'

    # image_list = glob.glob(equal_path + "*.png")
    # for im_name in image_list:
    head, tail = os.path.split(testImageFileName)
    # im = cv2.imread(testImageFileName)  # specify the image to process
    im = cv2.imread(testImageFileName, cv2.IMREAD_GRAYSCALE)
    MainImage2=copy.deepcopy(im)
    MainImage2[im < 120] = 255
    MainImage2[im >= 120] = 0
    rawRes =  initialBoxes(MainImage2)  # raw bounding boxes
    print(type(rawRes))
        # finalRes = connect(im, rawRes)  # connect i, division mark, equation mark, ellipsis
        # image_name = os.path.splitext(tail)[0]
        # saveImages(im, finalRes, image_name)


if __name__ == "__main__":
    main()