from keras.models import load_model
modelPath='C:\\Users\\sabik\\PycharmProjects\\DeepLearningBasics\\Model2.h5'
imagepath='C:\\Users\\sabik\\PycharmProjects\\DeepLearningBasics\\TrainTestImages\\Train\\'
filename= imagepath + '4\\4 (5).jpg'
classifier = load_model(modelPath)
classifier.get_config()
from keras.preprocessing import image
import numpy as np
count=0
for i in range(1,5917,1):
    filename = imagepath + '4\\4 ('+str(i)+').jpg'
    test_image = image.load_img(filename, target_size = (45, 45))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    if(result[0][10]>=0.9):
        count+=1
print(count)