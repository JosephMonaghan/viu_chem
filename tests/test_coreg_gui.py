from __future__ import annotations

import numpy as np
import pytest

from viu_chem.coreg_gui import _translate_affine_in_world


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
