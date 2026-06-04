from viu_chem import MSI_Process as msi
import matplotlib.pyplot as plt
import matplotlib as mpl
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
    title_axes = fig.axes[4:8]
    row_label_axes = fig.axes[8:10]
    assert [text.get_text() for axis in title_axes for text in axis.texts] == ["A1", "A2", "B1", "B2"]
    assert [text.get_text() for axis in row_label_axes for text in axis.texts] == ["A", "B"]
    assert all(title_ax.get_position().y0 >= image_ax.get_position().y1 for title_ax, image_ax in zip(title_axes, image_axes))
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
    title_axes = fig.axes[4:8]
    header_axes = fig.axes[8:10]
    assert [text.get_text() for axis in title_axes for text in axis.texts] == ["A1", "B1", "A2", "B2"]
    assert [text.get_text() for axis in header_axes for text in axis.texts] == ["A", "B"]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    first_title_top = title_axes[0].texts[0].get_window_extent(renderer).y1
    first_header_bottom = header_axes[0].texts[0].get_window_extent(renderer).y0
    assert first_header_bottom - first_title_top > 15
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
    title_axes = fig.axes[4:8]
    header_axes = fig.axes[8:10]
    row_label_axes = fig.axes[10:12]
    assert [text.get_text() for axis in row_label_axes for text in axis.texts] == ["Day 1", "Day 3"]
    assert [text.get_text() for axis in header_axes for text in axis.texts] == ["A", "B"]
    assert [text.get_text() for axis in title_axes for text in axis.texts] == [
        "Day1_A", "Day1_B", "Day3_A", "Day3_B",
    ]
    fig.canvas.draw()
    assert all(axis.get_position().y0 > image_axes[0].get_position().y1 for axis in header_axes)
    plt.close(fig)


def test_draw_grid_wraps_long_row_labels_into_balanced_lines():
    fig = msi.drawGrid(
        [np.ones((2, 2)) for _ in range(3)],
        groups=["Dual Treatment CD73i", "Vehicle Control", "CAIXi"],
        group_axis="row",
    )

    row_labels = [text.get_text() for axis in fig.axes for text in axis.texts]
    assert "Dual Treatment\nCD73i" in row_labels
    assert "Vehicle\nControl" in row_labels
    assert "CAIXi" in row_labels
    plt.close(fig)


def test_draw_grid_respects_primary_and_secondary_group_order():
    fig = msi.drawGrid(
        [np.ones((2, 2)) * value for value in range(1, 5)],
        names=["Day1_A", "Day3_B", "Day1_B", "Day3_A"],
        groups=["Day 1", "Day 3", "Day 1", "Day 3"],
        group_axis="row",
        secondary_groups=["A", "B", "B", "A"],
        group_order=["Day 3", "Day 1"],
        secondary_group_order=["B", "A"],
    )

    title_axes = fig.axes[4:8]
    header_axes = fig.axes[8:10]
    row_label_axes = fig.axes[10:12]
    assert [text.get_text() for axis in title_axes for text in axis.texts] == [
        "Day3_B", "Day3_A", "Day1_B", "Day1_A",
    ]
    assert [text.get_text() for axis in header_axes for text in axis.texts] == ["B", "A"]
    assert [text.get_text() for axis in row_label_axes for text in axis.texts] == ["Day 3", "Day 1"]
    plt.close(fig)


def test_draw_grid_rejects_unknown_group_order_labels():
    with pytest.raises(ValueError, match="unknown labels"):
        msi.drawGrid(
            [np.ones((2, 2))],
            groups=["Control"],
            group_order=["Treatment"],
        )


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


def test_draw_grid_global_color_scale_shares_lower_and_upper_limits():
    images = [
        np.arange(1, 101).reshape(10, 10),
        np.arange(101, 201).reshape(10, 10),
    ]

    fig = msi.drawGrid(images, dims=(1, 2), lower_cut_off=5, cut_off=95)

    pooled = np.concatenate([image.ravel() for image in images])
    expected_limits = (np.percentile(pooled, 5), np.percentile(pooled, 95))
    assert [axis.images[0].get_clim() for axis in fig.axes[:2]] == [
        expected_limits,
        expected_limits,
    ]
    plt.close(fig)


def test_draw_grid_lower_cutoff_can_be_configured():
    image = np.arange(1, 101).reshape(10, 10)

    fig = msi.drawGrid([image], lower_cut_off=20, cut_off=80)

    assert fig.axes[0].images[0].get_clim() == (
        pytest.approx(np.percentile(image, 20)),
        pytest.approx(np.percentile(image, 80)),
    )
    cbar_ax = next(axis for axis in fig.axes if axis.get_ylabel() == "Intensity")
    assert [tick.get_text() for tick in cbar_ax.get_yticklabels()] == ["21", "50", "80+"]
    assert len(cbar_ax.patches) == 1
    assert cbar_ax.texts[0].get_text() == "100"
    plt.close(fig)


def test_draw_grid_non_global_colorbar_uses_percentile_labels():
    images = [np.arange(1, 101).reshape(10, 10), np.arange(101, 201).reshape(10, 10)]

    fig = msi.drawGrid(images, dims=(1, 2), lower_cut_off=20, cut_off=80, color_scale="image")

    cbar_ax = next(axis for axis in fig.axes if axis.get_ylabel() == "Intensity")
    assert [tick.get_text() for tick in cbar_ax.get_yticklabels()] == ["0%", "50%", "100%"]
    assert len(cbar_ax.patches) == 1
    assert cbar_ax.texts[0].get_text() == "125%"
    plt.close(fig)


def test_draw_grid_relative_colorbar_reports_largest_scope_overflow():
    images = [
        np.arange(1, 101).reshape(10, 10),
        np.arange(1, 201, 2).reshape(10, 10),
    ]

    fig = msi.drawGrid(images, dims=(1, 2), cut_off=50, color_scale="image")

    cbar_ax = next(axis for axis in fig.axes if axis.get_ylabel() == "Intensity")
    expected_ratio = max(
        np.max(image) / np.percentile(image, 50)
        for image in images
    )
    assert cbar_ax.texts[0].get_text() == f"{expected_ratio:.0%}"
    assert cbar_ax.texts[0].get_color() == "white"
    assert cbar_ax.texts[0].get_position()[1] > max(
        point[1] for point in cbar_ax.patches[0].get_xy()
    )
    plt.close(fig)


def test_draw_grid_can_suppress_relative_overflow_ratio_label():
    image = np.arange(1, 101).reshape(10, 10)

    fig = msi.drawGrid([image], cut_off=50, color_scale="image", show_overflow_ratio=False)

    cbar_ax = next(axis for axis in fig.axes if axis.get_ylabel() == "Intensity")
    assert len(cbar_ax.patches) == 1
    assert len(cbar_ax.texts) == 0
    plt.close(fig)


def test_draw_grid_global_colorbar_reports_absolute_maximum_above_cap():
    image = np.linspace(0, 2_000_000, 100).reshape(10, 10)

    fig = msi.drawGrid([image], cut_off=80)

    cbar_ax = next(axis for axis in fig.axes if axis.get_ylabel() == "Intensity")
    assert cbar_ax.texts[0].get_text() == r"$2.00 \times 10^{6}$"
    assert cbar_ax.texts[0].get_position()[1] > max(
        point[1] for point in cbar_ax.patches[0].get_xy()
    )
    plt.close(fig)


def test_draw_grid_can_suppress_global_maximum_label():
    image = np.arange(1, 101).reshape(10, 10)

    fig = msi.drawGrid([image], show_overflow_ratio=False)

    cbar_ax = next(axis for axis in fig.axes if axis.get_ylabel() == "Intensity")
    assert len(cbar_ax.patches) == 1
    assert len(cbar_ax.texts) == 0
    plt.close(fig)


def test_draw_grid_global_colorbar_has_no_overflow_cap_at_maximum_cutoff():
    image = np.arange(1, 101).reshape(10, 10)

    fig = msi.drawGrid([image], lower_cut_off=0, cut_off=100)

    cbar_ax = next(axis for axis in fig.axes if axis.get_ylabel() == "Intensity")
    assert len(cbar_ax.patches) == 0
    plt.close(fig)


def test_draw_grid_defaults_to_zero_lower_limit():
    image = np.arange(1, 101).reshape(10, 10)

    fig = msi.drawGrid([image])

    assert fig.axes[0].images[0].get_clim()[0] == 0
    plt.close(fig)


def test_draw_grid_applies_colormap_to_images_colorbar_and_overflow_cap():
    image = np.arange(1, 101).reshape(10, 10)

    fig = msi.drawGrid([image], cmap="jet")

    image_ax = fig.axes[0]
    cbar_ax = next(axis for axis in fig.axes if axis.get_ylabel() == "Intensity")
    assert image_ax.images[0].get_cmap().name == "jet"
    assert cbar_ax.images[0].get_cmap().name == "jet"
    assert cbar_ax.patches[0].get_facecolor() == pytest.approx(mpl.colormaps["jet"](1.0))
    assert cbar_ax.patches[0].get_edgecolor()[3] == 0
    assert all(not spine.get_visible() for spine in cbar_ax.spines.values())
    assert cbar_ax.get_facecolor() == pytest.approx(mpl.colormaps["jet"](0.0))
    assert fig.get_facecolor() == pytest.approx(mpl.colormaps["jet"](0.0))
    assert image_ax.get_facecolor() == pytest.approx(mpl.colormaps["jet"](0.0))
    plt.close(fig)


def test_draw_grid_colorbar_uses_readable_scientific_notation():
    image = np.linspace(0, 2_000_000, 100).reshape(10, 10)

    fig = msi.drawGrid([image], cut_off=80)

    cbar_ax = next(axis for axis in fig.axes if axis.get_ylabel() == "Intensity")
    assert [tick.get_text() for tick in cbar_ax.get_yticklabels()] == [
        "0",
        r"$8.02 \times 10^{5}$",
        r"$1.60 \times 10^{6}+$",
    ]
    plt.close(fig)


def test_draw_grid_colorbar_ticks_are_smaller_than_label_and_outer_margins_are_tight():
    fig = msi.drawGrid([np.arange(1, 101).reshape(10, 10)])

    cbar_ax = next(axis for axis in fig.axes if axis.get_ylabel() == "Intensity")
    label_size = cbar_ax.yaxis.label.get_fontsize()
    assert all(tick.get_fontsize() == label_size - 2 for tick in cbar_ax.get_yticklabels())
    grid = cbar_ax.get_subplotspec().get_gridspec()
    assert grid.left == pytest.approx(0.03)
    assert grid.right == pytest.approx(0.94)
    plt.close(fig)


def test_draw_grid_uses_contrasting_text_for_bright_colormap_background():
    fig = msi.drawGrid(
        [np.arange(1, 101).reshape(10, 10)],
        title="Bright background",
        names=["Sample"],
        cmap="twilight",
    )

    cbar_ax = next(axis for axis in fig.axes if axis.get_ylabel() == "Intensity")
    assert fig._suptitle.get_color() == "black"
    assert cbar_ax.yaxis.label.get_color() == "black"
    assert all(tick.get_color() == "black" for tick in cbar_ax.get_yticklabels())
    assert all(text.get_color() == "black" for axis in fig.axes for text in axis.texts)
    plt.close(fig)


def test_draw_grid_zero_percentile_cutoff_falls_back_to_image_maximum():
    sparse_image = np.zeros((10, 10))
    sparse_image[0, 0] = 25

    fig = msi.drawGrid([sparse_image], cut_off=90, color_scale="image")

    assert fig.axes[0].images[0].get_clim()[1] == 25
    plt.close(fig)


def test_draw_grid_global_cutoff_uses_pooled_dataset_percentile():
    sparse_outlier = np.zeros((10, 10))
    sparse_outlier[0, 0] = 100
    dense_image = np.ones((10, 10)) * 10

    fig = msi.drawGrid([sparse_outlier, dense_image], dims=(1, 2), cut_off=80)

    expected_cutoff = np.percentile(np.concatenate(([100], np.full(100, 10))), 80)
    assert [axis.images[0].get_clim()[1] for axis in fig.axes[:2]] == [expected_cutoff, expected_cutoff]
    plt.close(fig)


def test_draw_grid_global_zero_percentile_falls_back_to_pooled_maximum():
    images = [np.zeros((10, 10)), np.zeros((10, 10))]
    images[1][0, 0] = 25

    fig = msi.drawGrid(images, dims=(1, 2), cut_off=80)

    assert [axis.images[0].get_clim()[1] for axis in fig.axes[:2]] == [25, 25]
    plt.close(fig)


def test_draw_grid_percentile_cutoff_ignores_zero_background():
    image = np.zeros((10, 10))
    image.ravel()[:20] = np.arange(1, 21)

    fig = msi.drawGrid([image], cut_off=80, color_scale="image")

    assert fig.axes[0].images[0].get_clim()[1] == pytest.approx(np.percentile(np.arange(1, 21), 80))
    plt.close(fig)


def test_draw_grid_can_include_zeros_in_percentile_cutoff():
    image = np.zeros((10, 10))
    image.ravel()[:20] = np.arange(1, 21)

    fig = msi.drawGrid([image], cut_off=80, color_scale="image", ignore_zeros=False)

    assert fig.axes[0].images[0].get_clim()[1] == pytest.approx(np.percentile(image, 80))
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

    assert [text.get_text() for axis in fig.axes[4:8] for text in axis.texts] == [
        "Day1_A", "Day1_B", "Day3_A", "Day3_B",
    ]
    plt.close(fig)


def test_draw_grid_rejects_scale_bar_without_physical_dimensions():
    with pytest.raises(ValueError, match="image_dimensions"):
        msi.drawGrid([np.ones((2, 2))], scale_bar=5)
