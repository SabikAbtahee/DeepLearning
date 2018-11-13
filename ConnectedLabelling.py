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

#Save X,Y postition according to left and right
def savePositionOfImages(leftToRightPosition,center,new_image,positions):
    # x=int((xhigh+xlow)/2)
    # y=int((yhigh+ylow)/2)
    x=leftToRightPosition
    y=center
    centerImage=[y,new_image]
    positions[x]=centerImage
    return positions

def showImages(usefullImages):

    # print(len(usefullImages))
    # print(type(usefullImages))
    for i in range(len(usefullImages)):
        # print(i[0])
        plt.imshow(usefullImages[i])
        plt.show()


def rotate(usefullImages):

    pix = np.array(usefullImages)

    rotated=[]
    for i in range(4):
        rotated.append(np.rot90(pix, i+1))
        new_image = Image.fromarray(rotated[i])
        plt.imshow(new_image)
        plt.show()




def deleteUnnecessaryImages(zoomedInImages):
    usefullImages=[]
    # usefullPositions={}
    # x=0
    for i in range(len(zoomedInImages)):
        width, height = zoomedInImages[i].size
        if(width>0 and height>0):
            usefullImages.append(zoomedInImages[i])
            # usefullPositions[x]=positions[i]
            # x+=1
    return usefullImages



def makeZoomInImages(positions):
    zoomedInImages=[]

    for key, value in sorted(positions.items()):
        # print(key,value[1])
        pix = np.array(value[1])
        u = np.argwhere(pix > 250)
        a = np.min(u, axis=0)
        b = np.max(u, axis=0)
        xlow = a[0]
        xhigh = b[0]
        ylow = a[1]
        yhigh = b[1]

        y = xlow - 100
        w = (xhigh - xlow) + 250
        x = ylow - 100
        h = (yhigh - ylow) + 250
        crop_MainImage = pix[y:y + h, x:x + w]
        new_image = Image.fromarray(crop_MainImage)
        new_image = new_image.convert("L")
        zoomedInImages.append(new_image)

    return zoomedInImages

#Get connected components and the x,y positions
def connectedComponents(image):

    image = image.astype( 'uint8' )
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)

    sizes = stats[1:,-1]; nb_components = nb_components - 1
    # print(type(sizes[0]))
    # print(sizes[0])
    # print(type(output))
    # print(output)
    # print(type(nb_components))
    # print(nb_components)
    min_size = 150
    MainImage3 = np.zeros(( output.shape ))
    # allImages=[]
    allnewImages={}
    positions={}
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            MainImage3[output == i + 1] = 255
            center=(np.where(MainImage3==255)[0][0])
            leftToRightPosition=(np.where(MainImage3 == 255)[1][0])
            # print(leftToRightPosition,center)
            new_image = Image.fromarray(MainImage3)
            new_image = new_image.convert("L")
            positions = savePositionOfImages(leftToRightPosition, center, new_image, positions)
            # plt.imshow(new_image)
            # plt.show()
            MainImage3[output == i + 1] = 0

            # allnewImages[leftToRightPosition]=new_image
            # allImages.append(new_image)
    # maxw=0
    # check=True
    # for key, value in sorted(positions.items()):
    #     print(key)
    #     if(check==True):
    #         maxw=value[0]
    #         check=False


    return positions

#ReadImage And Convert all black or white
def readImage(testImageFileName):
    MainImage = cv2.imread(testImageFileName,cv2.IMREAD_GRAYSCALE)

    MainImage2=copy.deepcopy(MainImage)
    MainImage2[MainImage < 110] = 255
    MainImage2[MainImage>= 110]=0
    # d = cv2.dilate(MainImage2, kernel, iterations=1)
    # m = cv2.morphologyEx(MainImage2, cv2.MORPH_OPEN, kernel)
    # print(np.where(MainImage2>120))
    # print(MainImage2[1395][1365])
    return MainImage2




def predictionsIndividual(usefullImages):
    for i in range(len(usefullImages)):
        rotatedImages=rotate(usefullImages[i])

    # showImages(usefullImages)
    # predict()

def  sortPositions(usefullPositions):
    xposition={}
    yposition={}
    listValue=[]
    for key, value in usefullPositions.items():
        xposition[key]=value[0]
        yposition[key]=value[1]

    sorted_x = sorted(xposition.items(), key=operator.itemgetter(1))

    # for key, value in sorted_x.items():
    #     listValue.append(key)
    #
    # print(listValue)
    return sorted_x


def reverseColor(usefullImages):
    allImages = []
    for i in range(len(usefullImages)):
        # print(type(usefullImages[i]))
        usefullImages[i]=np.array(usefullImages[i])
        # print(type(usefullImages[i]))
        MainImage2 = copy.deepcopy(usefullImages[i])
        MainImage2[usefullImages[i] < 110] = 255
        MainImage2[usefullImages[i] >= 110] = 0
        new_image = Image.fromarray(MainImage2)
        new_image = new_image.convert("L")

        allImages.append(new_image)

    return allImages
    # showImages(allImages)
#
# def resizeImages(usefullImages):
#



def main():
    testImagePath = 'C:\\Users\\sabik\\PycharmProjects\\DeepLearningBasics\\TestIMages\\'
    savetestImagePath = 'C:\\Users\\sabik\\PycharmProjects\\DeepLearningBasics\\TestIMages\\'
    testImageFileName = testImagePath + 'Equ1.jpg'

    MainImage2=readImage(testImageFileName)
    positions=connectedComponents(MainImage2)
    zoomedInImages = makeZoomInImages(positions)
    usefullImages=deleteUnnecessaryImages(zoomedInImages)
    # print(type(usefullImages))
    blackImages = reverseColor(usefullImages)
    # print(type(blackImages))
    PredictFromModel.ParseForPrediction(blackImages)

    # print(type(usefullImages[1]))
    # resizedImage=PredictFromModel.resizeImage(usefullImages[1])

    # sortedX=sortPositions(usefullPositions)
    #
    # showImages(usefullImages)

    # showImages(blackImages)
    # resizeImages(usefullImages)
    # PredictFromModel.predict(usefullImages[1])

    # predictionsIndividual(usefullImages[0])

if __name__ == "__main__":
    main()









