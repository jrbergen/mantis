"""Initializes objects required for image-related tests"""

import numpy as np
from numpy import ndarray

# Define test images of various shapes
float_img_dtype = np.float64
int_img_dtype = np.uint8


def assert_shape(arr: ndarray, expected_shape: tuple[int, ...]) -> None:
    assert arr.shape == expected_shape, f"Expected shape {expected_shape}, got shape {arr.shape!r}."


fv: tuple[float, float, float, float, float, float, float, float, float] = (
    0.0,
    0.128,
    0.256,
    0.384,
    0.512,
    0.064,
    0.768,
    0.896,
    1.0,
)
"""Floating point values used for test images."""
a, b, c, d, e, f, g, h, i = fv

iv: tuple[int, int, int, int, int, int, int, int, int] = (0, 32, 64, 96, 128, 160, 192, 224, 255)
"""8-bit integer values used for test images."""

ai, bi, ci, di, ei, fi, gi, hi, ii = iv


class ImgMock:
    def __init__(self):
        pass

    class Square:
        pass

    class NonSquare:
        pass

    def imgs_square(self) -> list[ndarray]:
        return [
            getattr(self.Square, aname)
            for aname in dir(self.Square)
            if not aname.startswith("_") and aname.isupper()
        ]

    def imgs_nonsquare(self) -> list[ndarray]:
        return [
            getattr(self, aname)
            for aname in dir(self)
            if aname != "Square" and not aname.startswith("_") and aname.isupper()
        ]


class MockImgsGreyscaleFloat(ImgMock):
    # Greyscale float images (Y by X by 1)
    class Square:
        IMG_331_F_GREY = np.array(
            [[[a], [b], [c]], [[d], [e], [f]], [[g], [h], [i]]],
            dtype=float_img_dtype,
        )
        IMG_221_F_GREY = np.array([[[a], [b]], [[c], [d]]], dtype=float_img_dtype)

    IMG_321_F_GREY = np.array([[[a], [b]], [[c], [d]], [[e], [f]]], dtype=float_img_dtype)
    IMG_231_F_GREY = np.array([[[a], [b], [c]], [[d], [e], [f]]], dtype=float_img_dtype)
    IMG_311_F_GREY = np.array([[[a]], [[b]], [[c]]], dtype=float_img_dtype)
    IMG_131_F_GREY = np.array([[[a], [b], [c]]], dtype=float_img_dtype)

    assert_shape(Square.IMG_331_F_GREY, (3, 3, 1))
    assert_shape(Square.IMG_221_F_GREY, (2, 2, 1))
    assert_shape(IMG_321_F_GREY, (3, 2, 1))
    assert_shape(IMG_231_F_GREY, (2, 3, 1))
    assert_shape(IMG_311_F_GREY, (3, 1, 1))
    assert_shape(IMG_131_F_GREY, (1, 3, 1))

    # Greyscale float images (Y by X)
    IMG_330_F_GREY = np.array([[a, b, c], [d, e, f], [g, h, i]], dtype=float_img_dtype)
    IMG_220_F_GREY = np.array([[a, b], [c, d]], dtype=float_img_dtype)
    IMG_320_F_GREY = np.array([[a, b], [c, d], [e, f]], dtype=float_img_dtype)
    IMG_230_F_GREY = np.array([[a, b, c], [d, e, f]], dtype=float_img_dtype)
    IMG_130_F_GREY = np.array([[a, b, c]], dtype=float_img_dtype)
    IMG_310_F_GREY = np.array([[a], [b], [c]], dtype=float_img_dtype)

    assert_shape(IMG_220_F_GREY, (2, 2))
    assert_shape(IMG_320_F_GREY, (3, 2))
    assert_shape(IMG_230_F_GREY, (2, 3))
    assert_shape(IMG_130_F_GREY, (1, 3))
    assert_shape(IMG_310_F_GREY, (3, 1))


class MockImgsGreyscaleInt(ImgMock):
    # Greyscale int images (Y by X by 1)
    class Square:
        IMG_331_I_GREY = np.array(
            [[[ai], [bi], [ci]], [[di], [ei], [fi]], [[gi], [hi], [ii]]],
            dtype=int_img_dtype,
        )
        IMG_221_I_GREY = np.array([[[ai], [bi]], [[ci], [di]]], dtype=int_img_dtype)

    IMG_321_I_GREY = np.array([[[ai], [bi]], [[ci], [di]], [[ei], [fi]]], dtype=int_img_dtype)
    IMG_231_I_GREY = np.array([[[ai], [bi], [ci]], [[di], [ei], [fi]]], dtype=int_img_dtype)
    IMG_311_I_GREY = np.array([[[ai]], [[bi]], [[ci]]], dtype=int_img_dtype)
    IMG_131_I_GREY = np.array([[[ai], [bi], [ci]]], dtype=int_img_dtype)

    assert_shape(Square.IMG_331_I_GREY, (3, 3, 1))
    assert_shape(Square.IMG_221_I_GREY, (2, 2, 1))
    assert_shape(IMG_321_I_GREY, (3, 2, 1))
    assert_shape(IMG_231_I_GREY, (2, 3, 1))
    assert_shape(IMG_311_I_GREY, (3, 1, 1))
    assert_shape(IMG_131_I_GREY, (1, 3, 1))

    # Greyscale int images (Y by X)
    IMG_330_I_GREY = np.array([[ai, bi, ci], [di, ei, fi], [gi, hi, ii]], dtype=int_img_dtype)
    IMG_220_I_GREY = np.array([[ai, bi], [ci, di]], dtype=int_img_dtype)
    IMG_320_I_GREY = np.array([[ai, bi], [ci, di], [ei, fi]], dtype=int_img_dtype)
    IMG_230_I_GREY = np.array([[ai, bi, ci], [di, ei, fi]], dtype=int_img_dtype)
    IMG_130_I_GREY = np.array([[ai, bi, ci]], dtype=int_img_dtype)
    IMG_310_I_GREY = np.array([[ai], [bi], [ci]], dtype=int_img_dtype)

    assert_shape(IMG_220_I_GREY, (2, 2))
    assert_shape(IMG_320_I_GREY, (3, 2))
    assert_shape(IMG_230_I_GREY, (2, 3))
    assert_shape(IMG_130_I_GREY, (1, 3))
    assert_shape(IMG_310_I_GREY, (3, 1))


class MockImgsColorFloat(ImgMock):
    # Color float images (Y by X by 3)
    class Square:
        IMG_333_F_COLOR = np.array(
            [
                [[c, a, a], [a, c, a], [a, a, c]],
                [[f, a, a], [a, f, a], [a, a, f]],
                [[i, a, a], [a, i, a], [a, a, i]],
            ],
            dtype=float_img_dtype,
        )

        IMG_223_F_COLOR = np.array(
            [[[c, a, a], [a, c, a]], [[f, a, a], [a, f, a]]],
            dtype=float_img_dtype,
        )

    IMG_233_F_COLOR = np.array(
        [
            [[c, a, a], [a, c, a], [a, a, c]],
            [[f, a, a], [a, f, a], [a, a, f]],
        ],
        dtype=float_img_dtype,
    )
    IMG_323_F_COLOR = np.array(
        [
            [[c, a, a], [a, c, a]],
            [[f, a, a], [a, f, a]],
            [[i, a, a], [a, i, a]],
        ],
        dtype=float_img_dtype,
    )
    IMG_133_F_COLOR = np.array([[[c, a, a], [a, c, a], [a, a, c]]], dtype=float_img_dtype)
    IMG_313_F_COLOR = np.array([[[c, a, a]], [[f, a, a]], [[i, a, a]]], dtype=float_img_dtype)

    assert_shape(Square.IMG_333_F_COLOR, (3, 3, 3))
    assert_shape(Square.IMG_223_F_COLOR, (2, 2, 3))
    assert_shape(IMG_323_F_COLOR, (3, 2, 3))
    assert_shape(IMG_233_F_COLOR, (2, 3, 3))
    assert_shape(IMG_313_F_COLOR, (3, 1, 3))
    assert_shape(IMG_133_F_COLOR, (1, 3, 3))


class MockImgsColorInt(ImgMock):
    # Color int images (Y by X by 3)
    class Square:
        IMG_333_I_COLOR = np.array(
            [
                [[ci, ai, ai], [ai, ci, ai], [ai, ai, ci]],
                [[fi, ai, ai], [ai, fi, ai], [ai, ai, fi]],
                [[ii, ai, ai], [ai, ii, ai], [ai, ai, ii]],
            ],
            dtype=float_img_dtype,
        )
        IMG_223_I_COLOR = np.array(
            [[[ci, ai, ai], [ai, ci, ai]], [[fi, ai, ai], [ai, fi, ai]]],
            dtype=float_img_dtype,
        )

    IMG_233_I_COLOR = np.array(
        [
            [[ci, ai, ai], [ai, ci, ai], [ai, ai, ci]],
            [[fi, ai, ai], [ai, fi, ai], [ai, ai, fi]],
        ],
        dtype=float_img_dtype,
    )
    IMG_323_I_COLOR = np.array(
        [
            [[ci, ai, ai], [ai, ci, ai]],
            [[fi, ai, ai], [ai, fi, ai]],
            [[ii, ai, ai], [ai, ii, ai]],
        ],
        dtype=float_img_dtype,
    )
    IMG_133_I_COLOR = np.array([[[ci, ai, ai], [ai, ci, ai], [ai, ai, ci]]], dtype=float_img_dtype)
    IMG_313_I_COLOR = np.array([[[ci, ai, ai]], [[fi, ai, ai]], [[ii, ai, ai]]], dtype=float_img_dtype)

    assert_shape(Square.IMG_333_I_COLOR, (3, 3, 3))
    assert_shape(Square.IMG_223_I_COLOR, (2, 2, 3))
    assert_shape(IMG_323_I_COLOR, (3, 2, 3))
    assert_shape(IMG_233_I_COLOR, (2, 3, 3))
    assert_shape(IMG_313_I_COLOR, (3, 1, 3))
    assert_shape(IMG_133_I_COLOR, (1, 3, 3))


class MockImgsColorFloat(ImgMock):
    # Color float images (Y by X by 3)
    class Square:
        IMG_333_F_COLOR = np.array(
            [
                [[c, a, a], [a, c, a], [a, a, c]],
                [[f, a, a], [a, f, a], [a, a, f]],
                [[i, a, a], [a, i, a], [a, a, i]],
            ],
            dtype=float_img_dtype,
        )

        IMG_223_F_COLOR = np.array(
            [[[c, a, a], [a, c, a]], [[f, a, a], [a, f, a]]],
            dtype=float_img_dtype,
        )

    IMG_233_F_COLOR = np.array(
        [
            [[c, a, a], [a, c, a], [a, a, c]],
            [[f, a, a], [a, f, a], [a, a, f]],
        ],
        dtype=float_img_dtype,
    )
    IMG_323_F_COLOR = np.array(
        [
            [[c, a, a], [a, c, a]],
            [[f, a, a], [a, f, a]],
            [[i, a, a], [a, i, a]],
        ],
        dtype=float_img_dtype,
    )
    IMG_133_F_COLOR = np.array([[[c, a, a], [a, c, a], [a, a, c]]], dtype=float_img_dtype)
    IMG_313_F_COLOR = np.array([[[c, a, a]], [[f, a, a]], [[i, a, a]]], dtype=float_img_dtype)

    assert_shape(Square.IMG_333_F_COLOR, (3, 3, 3))
    assert_shape(Square.IMG_223_F_COLOR, (2, 2, 3))
    assert_shape(IMG_323_F_COLOR, (3, 2, 3))
    assert_shape(IMG_233_F_COLOR, (2, 3, 3))
    assert_shape(IMG_313_F_COLOR, (3, 1, 3))
    assert_shape(IMG_133_F_COLOR, (1, 3, 3))


class MockImgsColorInt(ImgMock):
    # Color int images (Y by X by 3)
    class Square:
        IMG_333_I_COLOR = np.array(
            [
                [[ci, ai, ai], [ai, ci, ai], [ai, ai, ci]],
                [[fi, ai, ai], [ai, fi, ai], [ai, ai, fi]],
                [[ii, ai, ai], [ai, ii, ai], [ai, ai, ii]],
            ],
            dtype=float_img_dtype,
        )
        IMG_223_I_COLOR = np.array(
            [[[ci, ai, ai], [ai, ci, ai]], [[fi, ai, ai], [ai, fi, ai]]],
            dtype=float_img_dtype,
        )

    IMG_233_I_COLOR = np.array(
        [
            [[ci, ai, ai], [ai, ci, ai], [ai, ai, ci]],
            [[fi, ai, ai], [ai, fi, ai], [ai, ai, fi]],
        ],
        dtype=float_img_dtype,
    )
    IMG_323_I_COLOR = np.array(
        [
            [[ci, ai, ai], [ai, ci, ai]],
            [[fi, ai, ai], [ai, fi, ai]],
            [[ii, ai, ai], [ai, ii, ai]],
        ],
        dtype=float_img_dtype,
    )
    IMG_133_I_COLOR = np.array([[[ci, ai, ai], [ai, ci, ai], [ai, ai, ci]]], dtype=float_img_dtype)
    IMG_313_I_COLOR = np.array([[[ci, ai, ai]], [[fi, ai, ai]], [[ii, ai, ai]]], dtype=float_img_dtype)

    assert_shape(Square.IMG_333_I_COLOR, (3, 3, 3))
    assert_shape(Square.IMG_223_I_COLOR, (2, 2, 3))
    assert_shape(IMG_323_I_COLOR, (3, 2, 3))
    assert_shape(IMG_233_I_COLOR, (2, 3, 3))
    assert_shape(IMG_313_I_COLOR, (3, 1, 3))
    assert_shape(IMG_133_I_COLOR, (1, 3, 3))
