import shutil
import os
import random


def check_folder_existance(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
           raise




def split_data():

    trainSize=0.8
    testSize=0.2

    main_dir_src = 'C:\\Users\\sabik\\PycharmProjects\\DeepLearningBasics\\Images'
    main_dir_dst = 'C:\\Users\\sabik\\PycharmProjects\\DeepLearningBasics\\TrainTestImages'

    train='Train'
    test='Test'



    random_image_index=[]

    folders=os.listdir(main_dir_src)

    for f in folders:
        images= os.listdir(main_dir_src + "/" + f)
        numberOfImages=len(images)
        ts=int(len(images)*0.2)
        random_image_index=random.sample(range(0, numberOfImages), ts)

        source_path = ""
        destination_path = ""

        for index in range(len(images)):
            if (index in random_image_index):
                source_path = main_dir_src + "/" + f + "/" + images[index]
                destination_path = main_dir_dst + "/" + test + "/" + f + "/" + images[index]
            else:
                source_path = main_dir_src + "/" + f + "/" + images[index]
                destination_path = main_dir_dst + "/" + train + "/" + f + "/" + images[index]

            check_folder_existance(destination_path)
            shutil.move(source_path, destination_path)

        print("{} completed".format(f))


if __name__ == "__main__":
    split_data()