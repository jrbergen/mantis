import itertools
from typing import NamedTuple, Optional

import pytest
from numpy import ndarray
import numpy as np

from tmap_defectdetector.image.rotations import rotate_img_square, rotate_img

IMG00 = IMG00_ROT_90 = IMG00_ROT_180 = IMG00_ROT_270 = np.array([])  # type: ignore
IMG11 = IMG11_ROT_90 = IMG11_ROT_180 = IMG11_ROT_270 = np.array([1])

IMG22 = np.array([[1, 2], [3, 4]])
IMG22_ROT_90 = np.array([[2, 4], [1, 3]])
IMG22_ROT_180 = np.array([[4, 3], [2, 1]])
IMG22_ROT_270 = np.array([[3, 1], [4, 2]])

IMG33 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
IMG33_ROT_90 = np.array([[3, 6, 9], [2, 5, 8], [1, 4, 7]])
IMG33_ROT_180 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
IMG33_ROT_270 = np.array([[7, 4, 1], [8, 5, 2], [9, 6, 3]])

NON_SQUARE_0 = np.array([1, 2])
NON_SQUARE_1 = np.array([[1, 2, 3], [4, 5, 6]])
NON_SQUARE_2 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
NON_SQUARE_3 = np.array([[1, 2], [3, 4], [5, 6]])
NON_SQUARE_4 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
NON_SQUARES: tuple[ndarray, ...] = (NON_SQUARE_0, NON_SQUARE_1, NON_SQUARE_2, NON_SQUARE_3, NON_SQUARE_4)


class RotateTestParams(NamedTuple):
    img: ndarray
    degrees: int
    expected: Optional[ndarray] = None


class TestRotateImg:
    @pytest.mark.parametrize(
        "img, degrees, expected",
        (
            RotateTestParams(img=IMG22, degrees=90, expected=IMG22_ROT_90),
            RotateTestParams(img=IMG33, degrees=90, expected=IMG33_ROT_90),
            RotateTestParams(img=IMG22, degrees=180, expected=IMG22_ROT_180),
            RotateTestParams(img=IMG33, degrees=180, expected=IMG33_ROT_180),
            RotateTestParams(img=IMG22, degrees=270, expected=IMG22_ROT_270),
            RotateTestParams(img=IMG33, degrees=270, expected=IMG33_ROT_270),
        ),
        ids=[f"{ndeg} degree rotation {dim}x{dim}" for ndeg, dim in itertools.product((90, 180, 270), (2, 3))],
    )
    def test_rotate_img_deg(self, img, degrees, expected):
        np.testing.assert_equal(rotate_img(img, degrees), expected)

    @pytest.mark.parametrize(
        "img, degrees",
        (
            (1, 90),
            ([1, 2], 90),
            ((1, 2), 90),
            ("NotNdarray", 90),
        ),
    )
    def test_rotate_img_wrongtype_no_ndarray_error(self, img, degrees):
        with pytest.raises(TypeError):
            rotate_img(img, degrees)

    @pytest.mark.parametrize(
        "img, degrees, expected",
        (
            RotateTestParams(img=IMG11, degrees=90, expected=IMG11_ROT_90),
            RotateTestParams(img=IMG11, degrees=180, expected=IMG11_ROT_180),
            RotateTestParams(img=IMG11, degrees=270, expected=IMG11_ROT_270),
        ),
        ids=[f"Single 1x1 pixel image; {ndeg} degree rotation 1x1" for ndeg in (90, 180, 270)],
    )
    @pytest.mark.filterwarnings("ignore")
    def test_rotate_img_single_value(self, img, degrees, expected):
        np.testing.assert_equal(rotate_img(img, degrees), expected)

    @pytest.mark.parametrize(
        "img, degrees, expected",
        (
            RotateTestParams(img=IMG00, degrees=90, expected=IMG00_ROT_90),
            RotateTestParams(img=IMG00, degrees=180, expected=IMG00_ROT_180),
            RotateTestParams(img=IMG00, degrees=270, expected=IMG00_ROT_270),
        ),
        ids=[f"Empty 0x0 image; {ndeg} degree rotation" for ndeg in (90, 180, 270)],
    )
    @pytest.mark.filterwarnings("ignore")
    def test_rotate_img_empty(self, img, degrees, expected):
        np.testing.assert_equal(rotate_img(img, degrees), expected)

    def test_rotate_img_single_value_warning(self):
        with pytest.warns(UserWarning, match=r".*singular image.*"):
            rotate_img(IMG11, 90)

    def test_rotate_img_empty_warning(self):
        with pytest.warns(UserWarning, match=r".*empty image.*"):
            rotate_img(IMG00, 90)


class TestRotateImgSquare(TestRotateImg):
    @pytest.mark.parametrize(
        "img, degrees, expected",
        (
            tuple(
                RotateTestParams(img=non_square, degrees=deg)
                for non_square, deg in itertools.product(NON_SQUARES, (0, 90))
            )
        ),
        ids=[
            f"Should raise ValueError test: non-square matrix [shape={arr.shape}, deg={ndeg}, iter={curi}]"
            for curi, (ndeg, arr) in enumerate(itertools.product((0, 90), NON_SQUARES))
        ],
    )
    def test_rotate_img_square_nonsquare_error(self, img, degrees, expected):
        with pytest.raises(ValueError):
            rotate_img_square(img, degrees)
