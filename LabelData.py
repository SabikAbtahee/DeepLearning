# from keras.models import load_model
# # from Code.ConnectedLabelling import *
# import os
# modelPath='C:/Users/sabik/PycharmProjects/DeepLearningBasics/AdvancedModel2.h5'
# imagepath='C:/Users/sabik/PycharmProjects/DeepLearningBasics/TrainTestImages/Train/3/3 (1).jpg'
# # filename= imagepath + '9\\9 (5).jpg'
#
# classifier = load_model(modelPath)
# # classifier.get_config()
# from keras.preprocessing import image
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
#
# count=0
# count=np.zeros((51,), dtype=int)
# labels=['','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','',]
# # path, dirs, files = next(os.walk("C:/Users/sabik/PycharmProjects/DeepLearningBasics/TrainTestImages/Train/1/1 (1).jpg"))
# # file_count = len(files)
#
# # print(file_count)
#
# filename = imagepath
# test_image = image.load_img(filename, target_size = (45, 45,1))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis=0)
#
# result = classifier.predict_classes(test_image)
# print(result)
# # for i in range(1,100,1):
# #     filename = imagepath + '1/1 ('+str(i)+').jpg'
#     # type(filename)
#     # test_image = image.load_img(filename, target_size = (45, 45))
#     # plt.imshow(test_image)
#     # plt.show()
#     # k = np.array(test_image)
#     # print(k)
#     # print(k.shape)
#     # y = k.reshape(1,45, 45,3)
#     # print(y.shape)
#     # test_image = image.img_to_array(test_image)
#     # test_image = np.expand_dims(test_image, axis = 0)
#     # type(test_image)
#     # result = classifier.predict(test_image,batch_size=None, verbose=0, steps=None)
#     # result=classifier.predict_classes(test_image)
#     # print(result)
#     # print(type(result))
#     # print(result[0])
#     # print(len(result))
#     # print(len(result[0]))
#     # print(result[0])
#     # print(result[1][7])
#     # for i in range(50):
#     #     if(result[i][7]==1):
#     #         count[i]+=1
#     #
#     # _, j = np.unravel_index(result.argmax(), result.shape)
#     # max_value_index = j
#     # print(max_value_index)
#     # img = cv2.imread(filename)
#     # img = cv2.resize(img, (45, 45))
#     # img = np.reshape(img, [1, 45, 45, 3])
#     # classes = classifier.predict_classes(img)
#     # print(classes)
# # print(count)