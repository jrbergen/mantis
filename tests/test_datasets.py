from __future__ import annotations

from abc import ABC

import numpy as np
import pytest as pytest
from PIL import Image

from tmap_defectdetector.dataset_downloaders import DatasetDownloaderELPV
from tmap_defectdetector.datasets import ImageDataSet, AbstractDataSet, ImageDataSetELPV
from tmap_defectdetector.samples import SampleLabelsELPV


@pytest.fixture
def abstract_dset():
    return AbstractDataSet()


@pytest.fixture
def sample_labels_ELPV() -> SampleLabelsELPV:

    return SampleLabelsELPV.from_csv(csv_path=downloader.label_paths[0])

def image_dataset_ELPV() -> ImageDataSetELPV:


def download_elpv_dataset():
    downloader = DatasetDownloaderELPV()
    downloader.download()


def _gen_img(img_width: int = 255, img_height: int = 255) -> Image:
    """Generates random image"""
    return Image.fromarray(np.uint8(np.random.rand(img_width, img_height))).convert("RGB")


class TestAbstractDataSet:

    def test_amplify_data(self):

        with pytest.raises(NotImplementedError):
            abstract_dset.amplify_data()


class TestImageDataset(AbstractDataSet):




