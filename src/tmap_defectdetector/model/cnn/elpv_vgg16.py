from __future__ import annotations

from pathlib import Path

import psutil as psutil
import tensorflow as tf
from keras import Model
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense
from keras_preprocessing.image import DirectoryIterator, ImageDataGenerator
from tensorflow.python.keras.callbacks import History

from tmap_defectdetector import DIR_APP, DIR_TMP, DIR_DATASETS
from tmap_defectdetector.logger import log


def elpv_data_generators(
    train_dir: Path = Path(DIR_DATASETS, "dataset-elpv/categories/train"),
    validate_dir: Path = Path(DIR_DATASETS, "dataset-elpv/categories/validate"),
    tgt_size: tuple[int, int] = (128, 128),
    rescale: float = 1.0 / 255,
    shear_range: float = 0.4,
    zoom_range: float = 0.5,
    rotation_range: int = 30,
    batch_size: int = 32,
    shuffle: bool = True,
    class_mode: str = "categorical",
    classes: tuple[str, ...] = ("pass1", "minor_damage", "major_damage", "fail"),
    horizontal_flip: bool = True,
    vertical_flip: bool = True,
) -> tuple[DirectoryIterator, DirectoryIterator]:
    if not train_dir.exists():
        train_dir.mkdir(parents=True, exist_ok=True)
    if not validate_dir.exists():
        validate_dir.mkdir(parents=True, exist_ok=True)

    log.info("[yellow2]Generating training data[/]...")
    train_datagen = ImageDataGenerator(
        rescale=rescale,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        rotation_range=rotation_range,
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        class_mode=class_mode,
        target_size=tgt_size,
        # classes=classes,
    )

    log.info("[yellow2]Generating validation data[/]...")
    validation_datagen = ImageDataGenerator(
        rescale=rescale,
    )
    validation_generator = validation_datagen.flow_from_directory(
        validate_dir,
        shuffle=shuffle,
        batch_size=batch_size,
        class_mode=class_mode,
        target_size=tgt_size,
        classes=classes,
    )
    return train_generator, validation_generator


def elpv_vgg16(
    epochs: int = 50,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    pooling: str = "avg",
    weights: str = "imagenet",
    include_top: bool = True,
    num_categories: int = 4,
) -> History:
    vgg16_model = VGG16(
        pooling=pooling,
        weights=weights,
        include_top=include_top,
        input_shape=input_shape,
        classes=num_categories if weights != "imagenet" else 1000,
    )
    for layers in vgg16_model.layers:
        layers.trainable = False
    vgg_x = Flatten()(vgg16_model.layers[-1].output)
    vgg_x = Dense(128, activation="relu")(vgg_x)
    vgg_x = Dense(num_categories, activation="softmax")(vgg_x)
    vgg16_final_model = Model(vgg16_model.input, vgg_x)
    vgg16_final_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

    vgg16_filepath = "vgg_16_" + "-saved-model-{epoch:02d}-acc-{val_acc:.2f}.hdf5"
    vgg_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        vgg16_filepath, monitor="val_acc", verbose=1, save_best_only=True, mode="max"
    )
    vgg_early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5)

    train_generator, validation_generator = elpv_data_generators(
        tgt_size=tuple(input_shape[:2]),
    )
    vgg16_history = vgg16_final_model.fit(
        train_generator,
        use_multiprocessing=True,
        workers=12 if 12 > psutil.cpu_count() > 1 else psutil.cpu_count(),
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[vgg_checkpoint, vgg_early_stopping],
        verbose=1,
    )
    return vgg16_history


if __name__ == "__main__":
    h = elpv_vgg16()
