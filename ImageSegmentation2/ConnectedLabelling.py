import cv2
import numpy as np
import copy
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def undesired_objects ( image ,savetestImagePath):
    count=0
    image = image.astype( 'uint8' )
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    min_size = 180
    img3 = np.zeros(( output.shape ))

    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            count+=1
            img3[output == i + 1] = 255
            new_image = Image.fromarray(img3)
            new_image = new_image.convert("L")
            new_image.save(savetestImagePath + str(count) + ".png")
            img3[output == i + 1] = 0
            # plt.imshow( img3 )
            # plt.show()
    print(count)
testImagePath='C:\\Users\\sabik\\PycharmProjects\\DeepLearningBasics\\TestIMages\\'
savetestImagePath='C:\\Users\\sabik\\PycharmProjects\\DeepLearningBasics\\TestIMages\\'
testImageFileName=testImagePath+'Equ1.jpg'




img = cv2.imread(testImageFileName,cv2.IMREAD_GRAYSCALE)
img2=copy.deepcopy(img)
img2[img < 120] = 255
img2[img>=120]=0
undesired_objects(img2,savetestImagePath)













#
#
# plt.imshow(img2)
# plt.show()


# # img[img >= 180] = 0
# img2[img < 120] = 255
# img2[img>=120]=0
# img2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)[1]
# imgray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
# ret,thresh = cv2.threshold(imgray,127,255,0)
# im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# # cv2.drawContours(img, contours, -1, (0,255,0), 3)
# # cv2.drawContours(img, contours, 3, (0,255,0), 3)
# res = []
# for cnt in contours:
#     x, y, w, h = cv2.boundingRect(cnt)
#     # exclude the whole size image and noisy point
#     if x is 0: continue
#     if w * h < 25: continue
#
#     res.append([(x, y), (x + w, y + h)])
# # cnt = contours[4]
# # cv2.drawContours(img2, [cnt], 0, (0,255,0), 3)
# print(len(res))