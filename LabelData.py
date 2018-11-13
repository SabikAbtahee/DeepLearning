from keras.models import load_model
# from Code.ConnectedLabelling import *
modelPath='C:\\Users\\sabik\\PycharmProjects\\DeepLearningBasics\\Model4.h5'
imagepath='C:\\Users\\sabik\\PycharmProjects\\DeepLearningBasics\\TrainTestImages\\Train\\'
# filename= imagepath + '9\\9 (5).jpg'

classifier = load_model(modelPath)
# classifier.get_config()
from keras.preprocessing import image
import numpy as np

count=0
labels=['','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','',]

for i in range(1,200,1):
    filename = imagepath + '5\\5 ('+str(i)+').jpg'
    type(filename)
    test_image = image.load_img(filename, target_size = (45, 45))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    type(test_image)
    result = classifier.predict(test_image)
    print(result[0])
    # if(result[0][10]>=0.9):
    #     count+=1

print(count)