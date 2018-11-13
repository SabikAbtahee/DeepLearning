#
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras import initializers
#
#
# #
def makeModel():
    classifier = Sequential()
    classifier.add(Convolution2D(32,(3,3),input_shape=(45,45,3),activation='relu'))

    classifier.add(MaxPooling2D(pool_size=(2,2)))


    classifier.add(Convolution2D(64, (3, 3), activation='relu'))

    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())

    classifier.add(Dense(units= 64,activation='relu'))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(units=24,activation='softmax'))
    classifier.add(Dropout(rate=0.1))
    classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])       # try rmsprop
    return classifier
#
#
# def makeModel():
#     classifier = Sequential()
#     classifier.add(Conv2D(32, (3, 3), input_shape=(45, 45, 3), activation='relu',kernel_initializer=initializers.random_normal(stddev=0.04,mean = 0.00), bias_initializer = initializers.Constant(value=0.2)))
#     classifier.add(BatchNormalization(momentum=0.99))
#     classifier.add(MaxPooling2D(pool_size=(2, 2)))
#
#     classifier.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializers.random_normal(stddev=0.04,mean = 0.00), bias_initializer = initializers.Constant(value=0.2)))
#     classifier.add(BatchNormalization(momentum=0.99))
#     classifier.add(MaxPooling2D(pool_size=(2, 2)))
#
#     classifier.add(Conv2D(80, (2, 2), activation='relu', kernel_initializer=initializers.random_normal(stddev=0.04, mean=0.00),
#                bias_initializer=initializers.Constant(value=0.2)))
#     classifier.add(BatchNormalization(momentum=0.99))
#     classifier.add(MaxPooling2D(pool_size=(2, 2)))
#
#     classifier.add(Flatten())
#
#     classifier.add(Dense(512, activation='relu', kernel_initializer=initializers.random_normal(stddev=0.02, mean=0.00),
#                          bias_initializer=initializers.Constant(value=0.1)))
#     classifier.add(BatchNormalization(momentum=0.99))
#     classifier.add(
#         Dense(51, activation='softmax', kernel_initializer=initializers.random_normal(stddev=0.02, mean=0.00),
#               bias_initializer=initializers.Constant(value=0.1)))
#
#       # try rmsprop
#     return classifier
#
#
#
#
from keras_preprocessing.image import ImageDataGenerator
import sys
from PIL import Image
def trainTest(classifier):
    trainPath='C:\\Users\\sabik\\PycharmProjects\\DeepLearningBasics\\TrainTestImages\\Train'
    testPath='C:\\Users\\sabik\\PycharmProjects\\DeepLearningBasics\\TrainTestImages\\Test'
    savePath='C:\\Users\\sabik\\PycharmProjects\\DeepLearningBasics\\test2.h5'

    # train_datagen = ImageDataGenerator(
    #     rescale=1. / 255,
    #     shear_range=0.2,
    #     zoom_range=0.2,
    #     horizontal_flip=True)

    train_datagen=ImageDataGenerator(rescale = 1. / 255,
                                        rotation_range = 3,
                                        zoom_range = 0.1,
                                        fill_mode = 'nearest',
                                        horizontal_flip = True)
    # test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_datagen = ImageDataGenerator(rescale=1. / 255,
                                      rotation_range=3,
                                      zoom_range=0.1,
                                      fill_mode='nearest',
                                      horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        trainPath,
        target_size=(45, 45),
        batch_size=16,
        class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(
        testPath,
        target_size=(45, 45),
        batch_size=16,
        class_mode='categorical')

    print(train_generator.class_indices)
    #
    # classifier.fit_generator(
    #     train_generator,
    #     steps_per_epoch=75093/16,
    #     epochs=5,
    #     validation_data=test_generator,
    #     validation_steps=18773/16)
#
#     # 269493 / 64
#     # 67347 / 64
#     #
#     # classifier.summary()
#     #
#     # classifier.get_config()
    classifier.save(savePath)
#     # classifier.save('newTestingModel.h5')
#
#     validation_generator.class_indices
# def main():
classifier=makeModel()
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
trainTest(classifier)
#
#
# if __name__ == "__main__":
#     main()