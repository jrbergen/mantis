import numpy as np
import pytest
from numpy import ndarray

from mantis.image.mirrors import (
    mirror_horizontal,
    mirror_vertical,
    mirror_diag_bottomleft_topright,
    mirror_diag_topleft_bottomright,
)

IMG11 = IMG11_MIR_90 = IMG11_MIR_180 = IMG11_MIR_270 = np.array([1])

IMG22 = np.array([[1, 2], [3, 4]])
IMG22_MIR_HOR = np.array([[3, 4], [1, 2]])
IMG22_MIR_VER = np.array([[2, 1], [4, 3]])
IMG22_MIR_DIAG_TOPLEFT_BOTRIGHT = np.array([[1, 3], [2, 4]])
IMG22_MIR_DIAG_BOTLEFT_TOPRIGHT = np.array([[4, 2], [3, 1]])

IMG33 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
IMG33_MIR_HOR = np.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]])
IMG33_MIR_VER = np.array([[3, 2, 1], [6, 5, 4], [9, 8, 7]])
IMG33_MIR_DIAG_TOPLEFT_BOTRIGHT = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
IMG33_MIR_DIAG_BOTLEFT_TOPRIGHT = np.array([[9, 6, 3], [8, 5, 2], [7, 4, 1]])

NON_SQUARE_0 = np.array([1, 2])
NON_SQUARE_1 = np.array([[1, 2, 3], [4, 5, 6]])
NON_SQUARE_2 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
NON_SQUARE_3 = np.array([[1, 2], [3, 4], [5, 6]])
NON_SQUARE_4 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
NON_SQUARES: tuple[ndarray, ...] = (NON_SQUARE_0, NON_SQUARE_1, NON_SQUARE_2, NON_SQUARE_3, NON_SQUARE_4)


class TestImageMirrorFuncs:

    _NO_NDARRAY_TYPECHECK_OBJ = "Not a numpy array; wrong type should raise TypeError"

    def test_mirror_horizontal_2x2xN(self):
        """Horizontal mirroring of 2x2xN image array."""
        np.testing.assert_equal(mirror_horizontal(IMG22), IMG22_MIR_HOR)

    def test_mirror_horizontal_3x3xN(self):
        """Horizontal mirroring of 3x3xN image array."""
        np.testing.assert_equal(mirror_horizontal(IMG33), IMG33_MIR_HOR)

    def test_mirror_horizontal_error(self):
        """Error when input == something other than an np.ndarray instance for horizontal mirroring."""
        with pytest.raises(TypeError):
            mirror_horizontal(self._NO_NDARRAY_TYPECHECK_OBJ)

    def test_mirror_vertical_2x2xN(self):
        """Vertical mirroring of 2x2xN image array."""
        np.testing.assert_equal(mirror_vertical(IMG22), IMG22_MIR_VER)

    def test_mirror_vertical_3x3xN(self):
        """Vertical mirroring of 3x3xN image array."""
        np.testing.assert_equal(mirror_vertical(IMG33), IMG33_MIR_VER)

    def test_mirror_vertical_type_error(self):
        """Error when input == something other than an np.ndarray instance for vertical mirroring."""
        with pytest.raises(TypeError):
            mirror_vertical(self._NO_NDARRAY_TYPECHECK_OBJ)

    def test_mirror_diag_topleft_bottomright_2x2xN(self):
        """Diagonal mirroring of 2x2xN image array across the main diagonal."""
        np.testing.assert_equal(mirror_diag_topleft_bottomright(IMG22), IMG22_MIR_DIAG_TOPLEFT_BOTRIGHT)

    def test_mirror_diag_topleft_bottomright_3x3xN(self):
        """Diagonal mirroring of 3x3xN image array across the main diagonal."""
        np.testing.assert_equal(mirror_diag_topleft_bottomright(IMG33), IMG33_MIR_DIAG_TOPLEFT_BOTRIGHT)

    def test_mirror_diag_topleft_bottomright_type_error(self):
        """
        Error when input == something other than an np.ndarray instance
        for diagonal mirroring accross the main diagonal.
        """
        with pytest.raises(TypeError):
            mirror_diag_topleft_bottomright(self._NO_NDARRAY_TYPECHECK_OBJ)

    @pytest.mark.parametrize(
        "img",
        NON_SQUARES,
        ids=[
            f"Mirroring accross marjor diagonal -> non-square image w/ shape {ns.shape} ValueError check"
            for ns in NON_SQUARES
        ],
    )
    def test_mirror_diag_topleft_bottomright_nonsquare_error(self, img):
        """Error non-square image is passed diagonal mirroring accross the major diagonal."""
        with pytest.raises(ValueError):
            mirror_diag_topleft_bottomright(img)

    def test_mirror_diag_bottomleft_topright_2x2xN(self):
        """Diagonal mirroring of 2x2xN image array across the minor diagonal."""
        np.testing.assert_equal(mirror_diag_bottomleft_topright(IMG22), IMG22_MIR_DIAG_BOTLEFT_TOPRIGHT)

    def test_mirror_diag_bottomleft_topright_3x3xN(self):
        """Diagonal mirroring of 3x3xN image array across the minor diagonal."""
        np.testing.assert_equal(mirror_diag_bottomleft_topright(IMG33), IMG33_MIR_DIAG_BOTLEFT_TOPRIGHT)

    def test_mirror_diag_bottomleft_topright_type_error(self):
        """
        Error when input == something other than an np.ndarray instance
        for diagonal mirroring accross the minor diagonal.
        """
        with pytest.raises(TypeError):
            mirror_diag_bottomleft_topright(self._NO_NDARRAY_TYPECHECK_OBJ)

    @pytest.mark.parametrize(
        "img",
        NON_SQUARES,
        ids=[
            f"Mirroring accross minor diagonal -> non-square image w/ shape {ns.shape} ValueError check"
            for ns in NON_SQUARES
        ],
    )
    def test_mirror_diag_bottomleft_topright_nonsquare_error(self, img):
        """
        Error non-square image is passed diagonal mirroring accross the minor diagonal.
        """
        with pytest.raises(ValueError):
            mirror_diag_bottomleft_topright(img)
