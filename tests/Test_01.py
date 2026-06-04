from viu_chem import MSI_Process as msi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import warnings

DATAFILE = "tests/Datafiles/Demo__FTMS + p ESI Full ms [70.imzML"
CHECKSUM = 19665189341.802734
MULTI_CHECKSUM = 20731235074.742188
ASPECT = 4.645352069116426

def test_matrix():
    assert np.sum(msi.get_image_matrix(DATAFILE)) == CHECKSUM

def multi_img():
    img_matrix = msi.get_image_matrix(src=DATAFILE,mz=[104.1070, 137.0709])
    multi_sum = 0
    for img in img_matrix:
        multi_sum += np.sum(img)
    
    assert multi_sum == MULTI_CHECKSUM
        
def test_img_draw():
    img = msi.get_image_matrix(DATAFILE)
    aspect_ratio = msi.get_aspect_ratio(DATAFILE)
    msi.draw_ion_image(img,"magma",mode='save',path="tests/images/test.tif",asp=aspect_ratio)

def check_aspect():
    assert msi.get_aspect_ratio(DATAFILE) == ASPECT


def test_draw_grid_groups_samples_by_row():
    images = [np.ones((2, 2)) * value for value in range(1, 5)]

    fig = msi.drawGrid(
        images,
        names=["A1", "B1", "A2", "B2"],
        groups=["A", "B", "A", "B"],
        group_axis="row",
    )

    image_axes = fig.axes[:4]
    assert [axis.get_title() for axis in image_axes] == ["A1", "A2", "B1", "B2"]
    assert [text.get_text() for axis in image_axes for text in axis.texts] == ["A", "B"]
    plt.close(fig)


def test_draw_grid_groups_samples_by_column():
    images = [np.ones((2, 2)) * value for value in range(1, 5)]

    fig = msi.drawGrid(
        images,
        names=["A1", "B1", "A2", "B2"],
        groups=["A", "B", "A", "B"],
        group_axis="column",
    )

    image_axes = fig.axes[:4]
    assert [axis.get_title() for axis in image_axes] == ["A1", "B1", "A2", "B2"]
    assert [text.get_text() for axis in image_axes for text in axis.texts] == ["A", "B"]
    plt.close(fig)


def test_draw_grid_adds_secondary_group_labels():
    images = [np.ones((2, 2)) * value for value in range(1, 5)]

    fig = msi.drawGrid(
        images,
        names=["Day1_A", "Day3_B", "Day1_B", "Day3_A"],
        groups=["Day 1", "Day 3", "Day 1", "Day 3"],
        group_axis="row",
        secondary_groups=["A", "B", "B", "A"],
    )

    image_axes = fig.axes[:4]
    header_axes = fig.axes[4:6]
    assert [text.get_text() for axis in image_axes for text in axis.texts] == ["Day 1", "Day 3"]
    assert [text.get_text() for axis in header_axes for text in axis.texts] == ["A", "B"]
    assert [axis.get_title() for axis in image_axes] == ["Day1_A", "Day1_B", "Day3_A", "Day3_B"]
    fig.canvas.draw()
    assert all(axis.get_position().y0 > image_axes[0].get_position().y1 for axis in header_axes)
    plt.close(fig)


def test_draw_grid_rejects_inconsistent_secondary_group_labels():
    with pytest.raises(ValueError, match="pair must be unique"):
        msi.drawGrid(
            [np.ones((2, 2)) for _ in range(2)],
            groups=["Day 1", "Day 1"],
            group_axis="row",
            secondary_groups=["A", "A"],
        )


def test_draw_grid_centers_physical_dimensions_and_adds_scale_bar():
    fig = msi.drawGrid(
        [np.ones((2, 2)), np.ones((2, 2))],
        dims=(1, 2),
        image_dimensions=[(10, 20), (20, 10)],
        scale_bar=5,
        scale_units="mm",
    )

    first_image, second_image = fig.axes[0].images[0], fig.axes[1].images[0]
    assert first_image.get_extent() == [5, 15, 20, 0]
    assert second_image.get_extent() == [0, 20, 15, 5]
    assert fig.axes[0].get_xlim() == fig.axes[1].get_xlim() == (0, 20)
    assert np.mean(fig.axes[0].get_ylim()) == pytest.approx(10)
    assert np.allclose(fig.axes[0].get_ylim(), fig.axes[1].get_ylim())
    assert any(text.get_text() == "5 mm" for axis in fig.axes for text in axis.texts)
    assert fig.get_size_inches()[1] == pytest.approx(2)
    fig.canvas.draw()
    assert fig.axes[2].get_position().x0 > max(axis.get_position().x1 for axis in fig.axes[:2])
    plt.close(fig)


def test_draw_grid_physical_layout_and_scale_bar_respond_to_resize():
    fig = msi.drawGrid(
        [np.ones((2, 2))],
        dims=(1, 1),
        image_dimensions=[(20, 10)],
        scale_bar=5,
    )
    image_ax = fig.axes[0]
    scale_line = fig.axes[-1].lines[0]
    initial_limits = image_ax.get_xlim(), image_ax.get_ylim()

    width, height = fig.get_size_inches()
    fig.set_size_inches(width * 2, height)
    fig.canvas.draw()

    assert (image_ax.get_xlim(), image_ax.get_ylim()) != initial_limits
    scale_bar_pixels = np.ptp(
        scale_line.get_transform().transform(np.column_stack((scale_line.get_xdata(), [0, 0])))[:, 0]
    )
    physical_length_pixels = np.ptp(image_ax.transData.transform([(0, 0), (5, 0)])[:, 0])
    assert scale_bar_pixels == pytest.approx(physical_length_pixels)
    plt.close(fig)


def test_draw_grid_uses_physical_canvas_aspect_for_figure_height():
    fig = msi.drawGrid(
        [np.ones((2, 2)) for _ in range(2)],
        dims=(1, 2),
        image_dimensions=[(40, 10), (40, 10)],
    )

    assert fig.get_size_inches()[1] == pytest.approx(0.5)
    plt.close(fig)


def test_draw_grid_column_layout_draws_without_layout_warning():
    images = [np.ones((2, 2)) for _ in range(16)]
    groups = ["Day 1", "Day 3"] * 8
    secondary_groups = [compound for compound in "ABCDEFGH" for _ in range(2)]

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*layout.*")
        fig = msi.drawGrid(
            images,
            groups=groups,
            group_axis="column",
            secondary_groups=secondary_groups,
            image_dimensions=[(40, 20)] * len(images),
            scale_bar=5,
        )
        fig.canvas.draw()

    plt.close(fig)


@pytest.mark.parametrize(
    ("color_scale", "expected_limits"),
    [
        ("global", [4, 4, 4, 4]),
        ("row", [2, 2, 4, 4]),
        ("column", [3, 4, 3, 4]),
        ("image", [1, 2, 3, 4]),
        ("none", [1, 2, 3, 4]),
    ],
)
def test_draw_grid_color_scaling_scope(color_scale, expected_limits):
    images = [np.ones((2, 2)) * value for value in range(1, 5)]

    fig = msi.drawGrid(images, dims=(2, 2), color_scale=color_scale)

    assert [axis.images[0].get_clim()[1] for axis in fig.axes[:4]] == expected_limits
    plt.close(fig)


def test_draw_grid_zero_percentile_cutoff_falls_back_to_image_maximum():
    sparse_image = np.zeros((10, 10))
    sparse_image[0, 0] = 25

    fig = msi.drawGrid([sparse_image], cut_off=90, color_scale="image")

    assert fig.axes[0].images[0].get_clim()[1] == 25
    plt.close(fig)


def test_draw_grid_uses_narrow_right_labeled_colorbar():
    fig = msi.drawGrid([np.ones((2, 2))])
    image_ax = fig.axes[0]
    cbar_ax = next(axis for axis in fig.axes if axis.get_ylabel() == "Intensity")

    assert cbar_ax.get_position().width < image_ax.get_position().width / 2
    assert cbar_ax.yaxis.get_ticks_position() == "right"
    assert cbar_ax.yaxis.get_label_position() == "right"
    plt.close(fig)


def test_draw_grid_accepts_pandas_series_with_nondefault_indexes():
    index = [10, 20, 30, 40]
    images = pd.Series([np.ones((2, 2)) * value for value in range(1, 5)], index=index)
    groups = pd.Series(["Day 1", "Day 3", "Day 1", "Day 3"], index=index)
    secondary_groups = pd.Series(["A", "B", "B", "A"], index=index)
    names = pd.Series(["Day1_A", "Day3_B", "Day1_B", "Day3_A"], index=index)

    fig = msi.drawGrid(
        images=images,
        groups=groups,
        group_axis="row",
        secondary_groups=secondary_groups,
        names=names,
    )

    assert [axis.get_title() for axis in fig.axes[:4]] == ["Day1_A", "Day1_B", "Day3_A", "Day3_B"]
    plt.close(fig)


def test_draw_grid_rejects_scale_bar_without_physical_dimensions():
    with pytest.raises(ValueError, match="image_dimensions"):
        msi.drawGrid([np.ones((2, 2))], scale_bar=5)
