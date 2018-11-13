# import cv2
# # a=cv2.imread('TestIMages/Equ1.jpg',0)  #pass 0 to convert into gray level
# # print(a.shape)
# # ret,thr = cv2.threshold(a, 0, 255, cv2.THRESH_OTSU)
# # print(a.shape)
# # # cv2.imshow('win1', thr)
# # # cv2.waitKey(0)
# # # cv2.destroyAllWindows()


#
# import cv2
# im_gray = cv2.imread('TestIMages/Equ1.jpg', cv2.IMREAD_GRAYSCALE)
# (thresh, im_bw) = cv2.threshold(im_gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# thresh = 150
# # im_binary = (im_gray,127,255,cv2.THRESH_BINARY
# ret,thresh_img = cv2.threshold(im_gray,110,255,cv2.THRESH_BINARY)
# cv2.imwrite('binary_image.png', thresh_img)




import os, sys
import cv2

path = "C:/Users/sabik/PycharmProjects/DeepLearningBasics/TrainTestImages/Test/yA/" #Source

dirs = os.listdir(path)


y=0

for item in dirs:

        y = y + 1
        im_gray = cv2.imread(path+item)
        ret, thresh_img = cv2.threshold(im_gray, 110, 255, cv2.THRESH_BINARY)
        cv2.imwrite('C:/Users/sabik/PycharmProjects/DeepLearningBasics/TrainTestImages/Test/yB/' + '-' + y.__str__() + '.png', thresh_img) #destination


            # cv2.imwrite('binary_image.png', thresh_img)
            # im = Image.open(path+item)
            # f, e = os.path.splitext(path+item)
            # imResize = im.resize((200,200), Image.ANTIALIAS)
            # imResize.save(f + ' resized.jpg', 'JPEG', quality=90)

