"""Initializes objects required for image-related tests"""
import numpy as np


# Define test images of various shapes
from numpy import ndarray

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

# Greyscale float images (Y by X by 1)
IMG_331_f_grey = np.array(
    [[[a], [b], [c]], [[d], [e], [f]], [[g], [h], [i]]],
    dtype=float_img_dtype,
)
IMG_221_f_grey = np.array([[[a], [b]], [[c], [d]]], dtype=float_img_dtype)
IMG_321_f_grey = np.array([[[a], [b]], [[c], [d]], [[e], [f]]], dtype=float_img_dtype)
IMG_231_f_grey = np.array([[[a], [b], [c]], [[d], [e], [f]]], dtype=float_img_dtype)
IMG_311_f_grey = np.array([[[a]], [[b]], [[c]]], dtype=float_img_dtype)
IMG_131_f_grey = np.array([[[a], [b], [c]]], dtype=float_img_dtype)

assert_shape(IMG_331_f_grey, (3, 3, 1))
assert_shape(IMG_221_f_grey, (2, 2, 1))
assert_shape(IMG_321_f_grey, (3, 2, 1))
assert_shape(IMG_231_f_grey, (2, 3, 1))
assert_shape(IMG_311_f_grey, (3, 1, 1))
assert_shape(IMG_131_f_grey, (1, 3, 1))


# Greyscale float images (Y by X)
IMG_330_f_grey = np.array([[a, b, c], [d, e, f], [g, h, i]], dtype=float_img_dtype)
IMG_220_f_grey = np.array([[a, b], [c, d]], dtype=float_img_dtype)
IMG_320_f_grey = np.array([[a, b], [c, d], [e, f]], dtype=float_img_dtype)
IMG_230_f_grey = np.array([[a, b, c], [d, e, f]], dtype=float_img_dtype)
IMG_130_f_grey = np.array([[a, b, c]], dtype=float_img_dtype)
IMG_310_f_grey = np.array([[a], [b], [c]], dtype=float_img_dtype)

assert_shape(IMG_220_f_grey, (2, 2))
assert_shape(IMG_320_f_grey, (3, 2))
assert_shape(IMG_230_f_grey, (2, 3))
assert_shape(IMG_130_f_grey, (1, 3))
assert_shape(IMG_310_f_grey, (3, 1))


# Greyscale int images (Y by X by 1)
IMG_331_i_grey = np.array(
    [[[ai], [bi], [ci]], [[di], [ei], [fi]], [[gi], [hi], [ii]]],
    dtype=int_img_dtype,
)
IMG_221_i_grey = np.array([[[ai], [bi]], [[ci], [di]]], dtype=int_img_dtype)
IMG_321_i_grey = np.array([[[ai], [bi]], [[ci], [di]], [[ei], [fi]]], dtype=int_img_dtype)
IMG_231_i_grey = np.array([[[ai], [bi], [ci]], [[di], [ei], [fi]]], dtype=int_img_dtype)
IMG_311_i_grey = np.array([[[ai]], [[bi]], [[ci]]], dtype=int_img_dtype)
IMG_131_i_grey = np.array([[[ai], [bi], [ci]]], dtype=int_img_dtype)

assert_shape(IMG_331_i_grey, (3, 3, 1))
assert_shape(IMG_221_i_grey, (2, 2, 1))
assert_shape(IMG_321_i_grey, (3, 2, 1))
assert_shape(IMG_231_i_grey, (2, 3, 1))
assert_shape(IMG_311_i_grey, (3, 1, 1))
assert_shape(IMG_131_i_grey, (1, 3, 1))


# Greyscale int images (Y by X)
IMG_330_i_grey = np.array([[ai, bi, ci], [di, ei, fi], [gi, hi, ii]], dtype=int_img_dtype)
IMG_220_i_grey = np.array([[ai, bi], [ci, di]], dtype=int_img_dtype)
IMG_320_i_grey = np.array([[ai, bi], [ci, di], [ei, fi]], dtype=int_img_dtype)
IMG_230_i_grey = np.array([[ai, bi, ci], [di, ei, fi]], dtype=int_img_dtype)
IMG_130_i_grey = np.array([[ai, bi, ci]], dtype=int_img_dtype)
IMG_310_i_grey = np.array([[ai], [bi], [ci]], dtype=int_img_dtype)

assert_shape(IMG_220_i_grey, (2, 2))
assert_shape(IMG_320_i_grey, (3, 2))
assert_shape(IMG_230_i_grey, (2, 3))
assert_shape(IMG_130_i_grey, (1, 3))
assert_shape(IMG_310_i_grey, (3, 1))


# Color float images (Y by X by 3)
IMG_333_f_color = np.array(
    [
        [[c, a, a], [a, c, a], [a, a, c]],
        [[f, a, a], [a, f, a], [a, a, f]],
        [[i, a, a], [a, i, a], [a, a, i]],
    ],
    dtype=float_img_dtype,
)
IMG_223_f_color = np.array(
    [[[c, a, a], [a, c, a]], [[f, a, a], [a, f, a]]],
    dtype=float_img_dtype,
)
IMG_233_f_color = np.array(
    [
        [[c, a, a], [a, c, a], [a, a, c]],
        [[f, a, a], [a, f, a], [a, a, f]],
    ],
    dtype=float_img_dtype,
)
IMG_323_f_color = np.array(
    [
        [[c, a, a], [a, c, a]],
        [[f, a, a], [a, f, a]],
        [[i, a, a], [a, i, a]],
    ],
    dtype=float_img_dtype,
)
IMG_133_f_color = np.array([[[c, a, a], [a, c, a], [a, a, c]]], dtype=float_img_dtype)
IMG_313_f_color = np.array([[[c, a, a]], [[f, a, a]], [[i, a, a]]], dtype=float_img_dtype)

assert_shape(IMG_333_f_color, (3, 3, 3))
assert_shape(IMG_223_f_color, (2, 2, 3))
assert_shape(IMG_323_f_color, (3, 2, 3))
assert_shape(IMG_233_f_color, (2, 3, 3))
assert_shape(IMG_313_f_color, (3, 1, 3))
assert_shape(IMG_133_f_color, (1, 3, 3))

# Color int images (Y by X by 3)
IMG_333_i_color = np.array(
    [
        [[ci, ai, ai], [ai, ci, ai], [ai, ai, ci]],
        [[fi, ai, ai], [ai, fi, ai], [ai, ai, fi]],
        [[ii, ai, ai], [ai, ii, ai], [ai, ai, ii]],
    ],
    dtype=float_img_dtype,
)
IMG_223_i_color = np.array(
    [[[ci, ai, ai], [ai, ci, ai]], [[fi, ai, ai], [ai, fi, ai]]],
    dtype=float_img_dtype,
)
IMG_233_i_color = np.array(
    [
        [[ci, ai, ai], [ai, ci, ai], [ai, ai, ci]],
        [[fi, ai, ai], [ai, fi, ai], [ai, ai, fi]],
    ],
    dtype=float_img_dtype,
)
IMG_323_i_color = np.array(
    [
        [[ci, ai, ai], [ai, ci, ai]],
        [[fi, ai, ai], [ai, fi, ai]],
        [[ii, ai, ai], [ai, ii, ai]],
    ],
    dtype=float_img_dtype,
)
IMG_133_i_color = np.array([[[ci, ai, ai], [ai, ci, ai], [ai, ai, ci]]], dtype=float_img_dtype)
IMG_313_i_color = np.array([[[ci, ai, ai]], [[fi, ai, ai]], [[ii, ai, ai]]], dtype=float_img_dtype)

assert_shape(IMG_333_i_color, (3, 3, 3))
assert_shape(IMG_223_i_color, (2, 2, 3))
assert_shape(IMG_323_i_color, (3, 2, 3))
assert_shape(IMG_233_i_color, (2, 3, 3))
assert_shape(IMG_313_i_color, (3, 1, 3))
assert_shape(IMG_133_i_color, (1, 3, 3))
