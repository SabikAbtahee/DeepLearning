from keras.models import load_model
# from Code.ConnectedLabelling import *
modelPath='C:\\Users\\sabik\\PycharmProjects\\DeepLearningBasics\\MainModel1.h5'
imagepath='C:\\Users\\sabik\\PycharmProjects\\DeepLearningBasics\\TrainTestImages\\Train\\'
# filename= imagepath + '4\\4 (5).jpg'
#
# classifier = load_model(modelPath)
# classifier.get_config()
from keras.preprocessing import image
import numpy as np
#
# count=0
#
# for i in range(1,5917,1):
#     filename = imagepath + '4\\4 ('+str(5)+').jpg'
#     type(filename)
#     test_image = image.load_img(filename, target_size = (45, 45))
#     test_image = image.img_to_array(test_image)
#     test_image = np.expand_dims(test_image, axis = 0)
#     type(test_image)
#     result = classifier.predict(test_image)
#     result
#     if(result[0][10]>=0.9):
#         count+=1
#
# print(count)

import cv2
import sys
import numpy as np
import copy
import os
from PIL import Image
import matplotlib.pyplot as plt
# from PredictFromModel import predict
import PredictFromModel
import operator
from resizeimage import resizeimage

def ParseForPrediction(blackImages):
    classifier = load_model(modelPath)
    equation=""
    print(type(blackImages[5]))
    for i in range(len(blackImages)):
        # print(type(blackImages[i]))
        blackImages[i]=np.array(blackImages[i])
        targetImage=resizeImage(blackImages[i])
        # print(targetImage)
        # print(type(targetImage))
        predictedCharacter=predict(targetImage,classifier)
        # equation+=predictedCharacter

def predict(targetImage,classifier):
    test_image = image.img_to_array(targetImage)
    test_image = np.expand_dims(test_image, axis = 0)
    # print(type(test_image))
    result = classifier.predict(test_image)
    print(result)


    return "x"




def resizeImage(targetimage):
    size = (45, 45)
    targetimage = cv2.cvtColor(targetimage, cv2.COLOR_GRAY2RGB)
    new_image = Image.fromarray(targetimage)
    # new_image = new_image.convert("L")
    # new_image = cv2.cvtColor(new_image, cv2.COLOR_GRAY2RGB)
    resizedImage = resizeimage.resize_cover(new_image, size)
    # print(type(resizedImage))
    # print(resizedImage.size)
    return resizedImage