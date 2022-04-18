from __future__ import annotations

import numpy as np
import pytest as pytest
from PIL import Image

from tmap_defectdetector.datasets import ImageDataSet


def _gen_img() -> Image:
    return Image.fromarray(np.uint8(np.random.rand(255, 255))).convert("RGB")


class TestImageDataSet:

    FULL_DSET_SZ = 10
    HALF_DSET_SZ = 5
    images = [_gen_img() for _ in range(FULL_DSET_SZ)]
    mixed_type_images = [_gen_img() for _ in range(FULL_DSET_SZ - 2)] + ["not_an_image", 42]
    another_image = _gen_img()
    img_lst = images[:HALF_DSET_SZ]
    img_tup = tuple(images[HALF_DSET_SZ:])
    dset_a = ImageDataSet(data=img_lst)
    dset_b = ImageDataSet(data=img_tup)

    def test_amplify_data(self):
        assert False, "Test not yet implemented"

    def test_generate_mirrors(self):
        assert False, "Test not yet implemented"

    def test_generate_translation(self):
        assert False, "Test not yet implemented"

    def test_generate_rotation(self):
        assert False, "Test not yet implemented"

    def test__add__(self):
        assert self.HALF_DSET_SZ == round(
            self.FULL_DSET_SZ / 2
        ), "Faulty test parameters? FULL_DSET_SZ should be HALF_DSET_SZ*2"
        assert len(self.dset_a) == len(self.dset_b) == self.HALF_DSET_SZ
        assert len(self.dset_a + self.dset_b) == self.FULL_DSET_SZ

        assert len(self.dset_a) == len(
            self.dset_a.data
        ), "Data attribute isn't of same size as object's __len__ return value."
        assert len(self.dset_b) == len(
            self.dset_b.data
        ), "Data attribute isn't of same size as object's __len__ return value."

        initial_len_c = len(self.dset_a)
        dset_c = self.dset_a + self.another_image
        assert (
            len(dset_c) == initial_len_c + 1
        ), "Unexpected DataSet size after addition of PIL.Image instance"
        # noinspection PyAugmentAssignment
        dset_c = dset_c + [self.another_image]
        assert (
            len(dset_c) == initial_len_c + 2
        ), "Unexpected DataSet size after addition of list w/ 1 PIL.Image instance"
        dset_c += (self.another_image,)
        assert (
            len(dset_c) == initial_len_c + 3
        ), "Unexpected DataSet size after addition of tuple w/ 1 PIL.Image instance"

        with pytest.raises(TypeError):
            dset_c = (
                dset_c + {self.another_image},
                "Addition of set of images should raise TypeError",
            )

        with pytest.raises(TypeError):
            dset_c = (
                dset_c + {"testdictionary": self.another_image},
                "Addition of dictionary should raise TypeError",
            )
