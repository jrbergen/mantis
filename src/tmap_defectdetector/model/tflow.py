from __future__ import annotations

# import numpy as np
# import os
# import six.moves.urllib as urllib
# import sys
# import tarfile
# import tensorflow as tf
# import zipfile

# from collections import defaultdict
# from io import StringIO
# from matplotlib import pyplot as plt
# from PIL import Image

# sys.path.append("..")
# from object_detection.utils import ops as utils_ops

# from object_detection.utils import label_map_util
# from object_detection.utils import visualization_utils as vis_util

# MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = "http://download.tensorflow.org/models/object_detection/"

# PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# PATH_TO_LABELS = os.path.join('mscoco_label_map.pbtxt')

# NUM_CLASSES = 90

# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)

# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#     file_name = os.path.basename(file.name)
#     if 'frozen_inference_graph.pb' in file_name:
#         tar_file.extract(file, os.getcwd())

# detection_graph = tf.Graph()
# with detection_graph.as_default():
#     od_graph_def = tf.compat.v1.GraphDef()
#     with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
#         serialized_graph = fid.read()
#         od_graph_def.ParseFromString(serialized_graph)
#         tf.import_graph_def(od_graph_def, name='')

# label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
# categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
# category_index = label_map_util.create_category_index(categories)

# print("Reached end successfully!")
from abc import abstractmethod
from typing import TYPE_CHECKING

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame

from tmap_defectdetector.dataset.datasets import ImageDataSetELPV
from tmap_defectdetector.logger import log

if TYPE_CHECKING:
    from build.lib.tmap_defectdetector.dataset.dataset_configs import DataSetConfigELPV


class DefectDetectionModel:
    pass

    @classmethod
    @abstractmethod
    def from_dataset(cls, dataset: ImageDataSetELPV) -> CNNModel:
        return NotImplemented


class CNNModel(DefectDetectionModel):
    def __init__(self):
        self.training_data: DataFrame = pd.DataFrame()
        self.test_data: DataFrame = pd.DataFrame()

    @classmethod
    def from_dataset(cls, dataset: ImageDataSetELPV) -> CNNModel:
        raise NotImplementedError(f"{cls.__name__}'s factory method not yet implemented.")


class CNNModelELPV(CNNModel):
    def __init__(self):
        """
        This must be generalized later; we need a generalized implementation for all (image)
        datasets, but given the time constraint we just build a class specific for the ELPV dataset.
        """
        super().__init__()

    @classmethod
    def from_dataset(cls, dataset: ImageDataSetELPV, tolerance: float = 1e-8) -> CNNModelELPV:
        pass

    def full_run_from_dataset(self, dataset: ImageDataSetELPV, tolerance: float = 1e-8):
        cfg: DataSetConfigELPV = dataset.dataset_cfg

        quality_col = cfg.SCHEMA_FULL.PROBABILITY.name
        type_col = cfg.SCHEMA_FULL.TYPE.name
        img_col = cfg.SCHEMA_FULL.SAMPLE.name
        id_col = cfg.SCHEMA_FULL.LABEL_SAMPLE_ID.name
        data: DataFrame = dataset.data
        # del data[img_col]
        # labels: DataFrame = dataset.labels

        if len(set(dataset.labels[type_col])) != 1:
            raise ValueError("Training dataset for multiple types; not yet supported.")

        # classifications = {s: [] for s in set(f"class_{ii}_{labelvalue:.2f}" for labelvalue in enumerate())}

        img_train = data.sample(frac=0.65)
        img_test = pd.concat([data, img_train]).drop_duplicates(subset=[id_col], keep=False)

        assert (len(img_train) + len(img_test)) == len(data)

        imgs_pass_train = (
            img_train[np.isclose(img_train[quality_col], 1.0, atol=tolerance)].loc[:, img_col].to_list()
        )
        imgs_review_train = img_train[img_train[quality_col].gt(0.01)].loc[:, img_col].to_list()
        imgs_fail_train = (
            img_train[np.isclose(img_train[quality_col], 0.0, atol=tolerance)].loc[:, img_col].to_list()
        )

        labels_pass_train = (
            img_train[np.isclose(img_train[quality_col], 1.0, atol=tolerance)].loc[:, quality_col].to_list()
        )
        labels_review_train = img_train[img_train[quality_col].gt(0.01)].loc[:, quality_col].to_list()
        labels_fail_train = (
            img_train[np.isclose(img_train[quality_col], 0.0, atol=tolerance)].loc[:, quality_col].to_list()
        )

        train_images = img_train.loc[:, img_col].to_list()
        train_labels = img_train.loc[:, quality_col].to_list()

        # imgs_pass_test = (
        #     img_test[np.isclose(img_test[quality_col], 1.0, atol=tolerance)].loc[:, img_col].to_list()
        # )
        # imgs_review_test = img_test[img_test[quality_col].gt(0.01)].loc[:, img_col].to_list()
        # imgs_fail_test = (
        #     img_test[np.isclose(img_test[quality_col], 0.0, atol=tolerance)].loc[:, img_col].to_list()
        # )
        #
        # labels_pass_test = (
        #     img_test[np.isclose(img_test[quality_col], 1.0, atol=tolerance)].loc[:, quality_col].to_list()
        # )
        # labels_review_test = img_test[img_test[quality_col].gt(0.01)].loc[:, quality_col].to_list()
        # labels_fail_test = (
        #     img_test[np.isclose(img_test[quality_col], 0.0, atol=tolerance)].loc[:, quality_col].to_list()
        # )
        #
        test_images = img_test.loc[:, img_col].to_list()
        test_labels = img_test.loc[:, quality_col].to_list()

        imgshape = imgs_fail_train[0].shape

        class_names = ["Fail", "Review_A", "Review_B", "Pass"]

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=imgshape),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(10),
            ]
        )

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        model.fit(train_images, train_labels, epochs=10)

        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
        log("\nTest accuracy:", test_acc)

        probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
        predictions = probability_model.predict(test_images)
        log(predictions[0])
        log(np.argmax(predictions[0]))
        log(test_labels[0])

        def plot_image(i, predictions_array, true_label, img):
            true_label, img = true_label[i], img[i]
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])

            plt.imshow(img, cmap=plt.cm.binary)

            predicted_label = np.argmax(predictions_array)
            if predicted_label == true_label:
                color = "blue"
            else:
                color = "red"

            plt.xlabel(
                "{} {:2.0f}% ({})".format(
                    class_names[predicted_label], 100 * np.max(predictions_array), class_names[true_label]
                ),
                color=color,
            )

        def plot_value_array(i, predictions_array, true_label):
            true_label = true_label[i]
            plt.grid(False)
            plt.xticks(range(10))
            plt.yticks([])
            thisplot = plt.bar(range(10), predictions_array, color="#777777")
            plt.ylim([0, 1])
            predicted_label = np.argmax(predictions_array)

            thisplot[predicted_label].set_color("red")
            thisplot[true_label].set_color("blue")

        i = 0
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plot_image(i, predictions[i], test_labels, test_images)
        plt.subplot(1, 2, 2)
        plot_value_array(i, predictions[i], test_labels)
        plt.show()

        # train_images, train_labels), (test_images, test_labels) = fashion_mnist

        # images = np.array(dataset.images)


def test_tflow():
    fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist

    class_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # plt.figure(figsize=(10,10))
    # for i in range(25):
    #     plt.subplot(5,5,i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(train_images[i], cmap=plt.cm.binary)
    #     plt.xlabel(class_names[train_labels[i]])
    # plt.show()

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.fit(train_images, train_labels, epochs=10)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print("\nTest accuracy:", test_acc)

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)
    print(predictions[0])
    print(np.argmax(predictions[0]))
    print(test_labels[0])

    def plot_image(i, predictions_array, true_label, img):
        true_label, img = true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary)

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = "blue"
        else:
            color = "red"

        plt.xlabel(
            "{} {:2.0f}% ({})".format(
                class_names[predicted_label], 100 * np.max(predictions_array), class_names[true_label]
            ),
            color=color,
        )

    def plot_value_array(i, predictions_array, true_label):
        true_label = true_label[i]
        plt.grid(False)
        plt.xticks(range(10))
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color("red")
        thisplot[true_label].set_color("blue")

    i = 0
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions[i], test_labels)
    plt.show()
