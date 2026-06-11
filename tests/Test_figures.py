import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PathCollection

from viu_chem import Figures


def test_unpack_dataframe_builds_nested_plot_data():
    data = pd.DataFrame({
        "approach": ["clipped", "clipped", "regular", "regular"],
        "attempt": [1, 2, 1, 2],
        "slope": [1.1, 1.2, 5.6, 11.5],
    })

    plot_data = Figures.unpack_dataframe(
        data,
        value="slope",
        primary_group="approach",
        secondary_group="attempt",
    )

    assert plot_data == {
        "clipped": {1: [1.1], 2: [1.2]},
        "regular": {1: [5.6], 2: [11.5]},
    }


def test_barchart_accepts_dataframe_group_columns():
    data = pd.DataFrame({
        "approach": ["clipped", "clipped", "regular", "regular"],
        "attempt": [1, 2, 1, 2],
        "slope": [1.1, 1.2, 5.6, 11.5],
    })

    fig, ax = plt.subplots()
    Figures.barchart(
        data,
        ax=ax,
        value="slope",
        primary_group="approach",
        secondary_group="attempt",
    )

    assert len(ax.patches) == 4
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["clipped", "regular"]
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["1", "2"]
    plt.close(fig)


def test_boxplot_accepts_dataframe_group_columns():
    data = pd.DataFrame({
        "approach": ["clipped", "clipped", "regular", "regular"],
        "attempt": [1, 2, 1, 2],
        "slope": [1.1, 1.2, 5.6, 11.5],
    })

    fig, ax = plt.subplots()
    Figures.boxplot(
        data,
        ax=ax,
        value="slope",
        primary_group="approach",
        secondary_group="attempt",
    )

    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["clipped", "regular"]
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["1", "2"]
    plt.close(fig)


def test_barchart_plots_grouped_bars_points_and_errorbars():
    data = {
        "Metabolite A": {
            "Low": [1, 2, 3],
            "High": [2, 4, 6],
        },
        "Metabolite B": {
            "Low": [2, 4, 6],
            "High": [3, 6, 9],
        },
    }

    fig, ax = plt.subplots()
    returned_ax = Figures.barchart(data, ax=ax, colors=["#2DB30C", "#8B19AA"])

    assert returned_ax is ax
    assert len(ax.patches) == 4
    assert [patch.get_height() for patch in ax.patches] == [2, 4, 4, 6]
    assert len([collection for collection in ax.collections if isinstance(collection, PathCollection)]) == 4
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["Low", "High"]
    plt.close(fig)


def test_barchart_plots_ungrouped_bars_and_points():
    data = {
        "Control": [1, 2, 3],
        "Treatment": [2, 4, 6],
    }

    fig, ax = plt.subplots()
    Figures.barchart(data, ax=ax, error="sd")

    assert len(ax.patches) == 2
    assert [patch.get_height() for patch in ax.patches] == [2, 4]
    assert len([collection for collection in ax.collections if isinstance(collection, PathCollection)]) == 2
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["Control", "Treatment"]
    plt.close(fig)


def test_barchart_defaults_to_standard_deviation_errorbars():
    data = {"Control": [1, 2, 3]}

    fig, ax = plt.subplots()
    Figures.barchart(data, ax=ax)

    error_cap_ys = sorted(float(line.get_ydata()[0]) for line in ax.lines)
    assert np.allclose(error_cap_ys, [1, 3])
    plt.close(fig)


def test_barchart_keeps_many_grouped_bars_nearly_touching_without_overlap():
    data = {
        "Sample": {
            str(idx): [idx, idx + 1, idx + 2]
            for idx in range(1, 6)
        }
    }

    fig, ax = plt.subplots()
    Figures.barchart(data, ax=ax)

    gaps = [
        right.get_x() - (left.get_x() + left.get_width())
        for left, right in zip(ax.patches, ax.patches[1:])
    ]
    assert min(gaps) > -1e-10
    assert max(gaps) < 0.01
    plt.close(fig)
