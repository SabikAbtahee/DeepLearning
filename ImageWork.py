import cv2
import matplotlib.pyplot as plt
from PIL import Image
def takeInput():
    ImagePath='TestIMages/Equ1.jpg'
    MainImage = cv2.imread(ImagePath, cv2.IMREAD_GRAYSCALE)
    MainImage=convertTograyScale(MainImage)
    plt.imshow(MainImage)
    plt.show()


def convertTograyScale(givenImage):
    new_image = Image.fromarray(givenImage)
    new_image = new_image.convert('LA')
    return new_image


def segmentIndividualCharacters(givenImagePath):
    im = cv2.imread(givenImagePath)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
    ret, im_th = cv2.threshold(im_gray, 110, 255, cv2.THRESH_BINARY_INV)
    _, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    images = {}
    x = 0
    for rect in rects:
        print(rect[0])
        # Draw the rectangles
        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
        # Make the rectangular region around the digit
        # print(type(rect[0]))
        leng = int(rect[3] * 1.3)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im_th[pt1:pt1 + leng, pt2:pt2 + leng]
        images[rect[0]] = roi
        # plt.imshow(roi)
        cv2.imshow('hello', roi)
        cv2.waitKey(0)
        x = x + 1
        # cv2.imwrite('d' + x.__str__() + '.jpg', roi)
        # Resize the image
        roi = cv2.resize(roi, (45, 45), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))


    # cv2.imshow("Resulting Image with Rectangular ROIs", im)
    # cv2.waitKey()
    y = 0
    for key in sorted(images):
        y = y + 1
        cv2.imwrite('y' + y.__str__() + '.jpg', images[key])
        # print(key,images[key])