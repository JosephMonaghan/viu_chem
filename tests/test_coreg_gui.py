from __future__ import annotations

import numpy as np
import pytest

from viu_chem.coreg_gui import (
    _limit_reference_pixels,
    _reference_level_alignment_to_world,
    _reference_pixel_stride,
    _reference_spatial_shape,
    _translate_affine_in_world,
)


def test_translate_affine_in_world_preserves_linear_component():
    transform_xy = np.array(
        [[2.0, 0.25, 5.0], [-0.5, 3.0, 7.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )

    translated = _translate_affine_in_world(transform_xy, (4.0, -6.0))

    np.testing.assert_allclose(translated[:, :2], transform_xy[:, :2])
    np.testing.assert_allclose(translated[:2, 2], [-1.0, 11.0])
    np.testing.assert_allclose(transform_xy[:2, 2], [5.0, 7.0])


@pytest.mark.parametrize("delta_yx", [(1.0,), (1.0, np.nan), (1.0, 2.0, 3.0)])
def test_translate_affine_in_world_rejects_invalid_drag_delta(delta_yx):
    with pytest.raises(ValueError, match="finite y and x"):
        _translate_affine_in_world(np.eye(3), delta_yx)


def test_reference_pyramid_alignment_is_composed_in_level_zero_world_coordinates():
    candidate_level_xy = np.array(
        [[1.0, 0.0, 3.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    fixed_to_world_xy = np.array(
        [[0.5, 0.0, 10.0], [0.0, 0.25, 20.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )

    result = _reference_level_alignment_to_world(
        candidate_level_xy,
        fixed_to_world_xy,
        moving_level_zero_shape=(800, 1200),
        moving_level_shape=(200, 300),
        fixed_level_zero_shape=(1000, 1600),
        fixed_level_shape=(250, 400),
    )

    expected = fixed_to_world_xy @ np.diag([4.0, 4.0, 1.0]) @ candidate_level_xy @ np.diag([0.25, 0.25, 1.0])
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [((100, 200), (100, 200)), ((100, 200, 3), (100, 200)), ((4, 100, 200), (100, 200))],
)
def test_reference_spatial_shape_handles_grayscale_and_rgb_without_materializing(shape, expected):
    class ShapeOnly:
        def __init__(self, value):
            self.shape = value

        def __array__(self):
            raise AssertionError("shape inspection must not materialize image data")

    assert _reference_spatial_shape(ShapeOnly(shape)) == expected


def test_reference_registration_pixel_budget_subsamples_before_materialization():
    image = np.zeros((2400, 3600, 3), dtype=np.uint8)

    limited = _limit_reference_pixels(image, 1_000_000)

    assert limited.shape[0] * limited.shape[1] <= 1_000_000
    assert limited.shape[2] == 3
    assert _reference_pixel_stride(image, 1_000_000) == 3
