
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout





def makeModel():
    classifier = Sequential()
    classifier.add(Convolution2D(32,(3,3),input_shape=(45,45,3),activation='relu'))

    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Convolution2D(64, (3, 3), activation='relu'))

    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())

    classifier.add(Dense(units= 64,activation='relu'))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(units=51,activation='softmax'))
    classifier.add(Dropout(rate=0.1))
    classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])       # try rmsprop
    return classifier

from keras_preprocessing.image import ImageDataGenerator
import sys
from PIL import Image
def trainTest(classifier):
    trainPath='C:\\Users\\sabik\\PycharmProjects\\DeepLearningBasics\\TrainTestImages\\Train'
    testPath='C:\\Users\\sabik\\PycharmProjects\\DeepLearningBasics\\TrainTestImages\\Test'
    savePath='C:\\Users\\sabik\\PycharmProjects\\DeepLearningBasics\\MainModel1.h5'

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        trainPath,
        target_size=(45, 45),
        batch_size=64,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        testPath,
        target_size=(45, 45),
        batch_size=64,
        class_mode='categorical')

    classifier.fit_generator(
        train_generator,
        steps_per_epoch=150000,
        epochs=4,
        validation_data=validation_generator,
        validation_steps=50000)

    classifier.summary()

    classifier.get_config()
    classifier.save(savePath)

def main():
    classifier=makeModel()
    trainTest(classifier)


if __name__ == "__main__":
    main()