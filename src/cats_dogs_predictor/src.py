from venv import create
import numpy as np #pip install numpy
import matplotlib.pyplot as plt #pip install matplotlib
import cv2 #pip install opencv-contrib-python
import os
import random
import pickle

DATADIR = "PetImages"
PICKLE_DIR = "pickle_files"
CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 50

training_data = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) #path to images
        class_num = CATEGORIES.index(category)
        # IMG_SIZE = 50
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

    # print(len(training_data))

    random.shuffle(training_data)
    for sample in training_data[:10]:
        print(sample[1])
    x = []
    y = []
    for features, label in training_data:
        x.append(features)
        y.append(label)
    x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    pickle_out = open("pickle_files/x.pickle", "wb")
    pickle.dump(x, pickle_out)
    pickle_out.close()

    pickle_out = open("pickle_files/y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()

def read_training_data():
    pickle_in = open("pickle_files/x.pickle", "rb")
    x = pickle.load(pickle_in)
    # print(x)

    pickle_in = open("pickle_files/y.pickle", "rb")
    y = pickle.load(pickle_in)
    # print(y)

# create_training_data()
# read_training_data()
