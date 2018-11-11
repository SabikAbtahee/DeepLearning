import cv2
import numpy as np
import copy
from PIL import Image
import matplotlib.pyplot as plt

def showImages(usefullImages):
    # print(len(usefullImages))
    for i in range(len(usefullImages)):
        plt.imshow(usefullImages[i])
        plt.show()
    # plt.imshow(usefullImages[i])
    # plt.show()
def rotate(usefullImages):
    pix = np.array(usefullImages[0])
    rotated = np.rot90(pix, 1)
    print(rotated)
    new_image = Image.fromarray(rotated)
    new_image = new_image.convert("L")

def deleteUnnecessaryImages(zoomedInImages):
    usefullImages=[]
    for i in range(len(zoomedInImages)):
        width, height = zoomedInImages[i].size
        if(width>0 and height>0):
            usefullImages.append(zoomedInImages[i])
    return usefullImages



def makeZoomInImages(allImages):
    zoomedInImages=[]
    for i in range(len(allImages)):
        pix = np.array(allImages[i])
        u=np.argwhere(pix>250)
        a=np.min(u,axis=0)
        b=np.max(u,axis=0)
        xlow=a[0]
        xhigh=b[0]
        ylow=a[1]
        yhigh=b[1]
        y=xlow-100
        w=(xhigh-xlow)+200
        x=ylow-100
        h=(yhigh-ylow)+250
        crop_MainImage = pix[y:y+h, x:x+w]
        new_image = Image.fromarray(crop_MainImage)
        new_image = new_image.convert("L")
        zoomedInImages.append(new_image)

    return zoomedInImages

def connectedComponents(image):

    image = image.astype( 'uint8' )
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    min_size = 180
    MainImage3 = np.zeros(( output.shape ))
    allImages=[]
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            MainImage3[output == i + 1] = 255
            new_image = Image.fromarray(MainImage3)
            new_image = new_image.convert("L")
            MainImage3[output == i + 1] = 0
            allImages.append(new_image)


    return allImages

def readImage(testImageFileName):
    MainImage = cv2.imread(testImageFileName,cv2.IMREAD_GRAYSCALE)
    MainImage2=copy.deepcopy(MainImage)
    MainImage2[MainImage < 120] = 255
    MainImage2[MainImage>=120]=0

    return MainImage2


def main():
    testImagePath = 'C:\\Users\\sabik\\PycharmProjects\\DeepLearningBasics\\TestIMages\\'
    savetestImagePath = 'C:\\Users\\sabik\\PycharmProjects\\DeepLearningBasics\\TestIMages\\'
    testImageFileName = testImagePath + 'Equ1.jpg'

    MainImage2=readImage(testImageFileName)
    allImages=connectedComponents(MainImage2)
    zoomedInImages = makeZoomInImages(allImages)
    usefullImages=deleteUnnecessaryImages(zoomedInImages)
    rotatedImages=rotate(usefullImages)
    # showImages(usefullImages)

if __name__ == "__main__":
    main()









