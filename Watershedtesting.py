import cv2
a=cv2.imread('TestIMages/Equ1.jpg',0)  #pass 0 to convert into gray level
print(a.shape)
ret,thr = cv2.threshold(a, 0, 255, cv2.THRESH_OTSU)
print(a.shape)
# cv2.imshow('win1', thr)
# cv2.waitKey(0)
# cv2.destroyAllWindows()