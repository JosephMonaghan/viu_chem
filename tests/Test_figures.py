import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PathCollection
from scipy.stats import false_discovery_control, ttest_ind, ttest_rel

from viu_chem import Figures


def test_volcano_uses_paired_t_test_when_requested():
    numerator = pd.DataFrame(
        [[11.0, 22.0, 31.0, 42.0]],
        index=[100.0],
    )
    denominator = pd.DataFrame(
        [[10.0, 20.0, 30.0, 40.0]],
        index=[100.0],
    )

    independent_fig, independent_ax = plt.subplots()
    _, independent_results = Figures.volcano(
        numerator,
        denominator,
        ax=independent_ax,
    )
    paired_fig, paired_ax = plt.subplots()
    _, paired_results = Figures.volcano(
        numerator,
        denominator,
        ax=paired_ax,
        paired=True,
    )

    expected_independent = ttest_ind(
        numerator.iloc[0], denominator.iloc[0], equal_var=False
    ).pvalue
    expected_paired = ttest_rel(
        np.log2(numerator.iloc[0]),
        np.log2(denominator.iloc[0]),
    ).pvalue
    expected_paired_fold_change = np.mean(
        np.log2(numerator.iloc[0].to_numpy() / denominator.iloc[0].to_numpy())
    )
    assert np.isclose(independent_results.loc[100.0, "pval"], expected_independent)
    assert np.isclose(paired_results.loc[100.0, "pval"], expected_paired)
    assert paired_results.loc[100.0, "pval"] < independent_results.loc[100.0, "pval"]
    assert np.isclose(paired_results.loc[100.0, "fold_change"], expected_paired_fold_change)
    assert not np.isclose(
        paired_results.loc[100.0, "fold_change"],
        independent_results.loc[100.0, "fold_change"],
    )
    plt.close(independent_fig)
    plt.close(paired_fig)


def test_volcano_rejects_unequal_sample_counts_for_paired_test():
    numerator = pd.DataFrame([[1.0, 2.0, 3.0]], index=[100.0])
    denominator = pd.DataFrame([[1.0, 2.0]], index=[100.0])

    with np.testing.assert_raises_regex(
        ValueError, "same number of samples"
    ):
        Figures.volcano(numerator, denominator, paired=True)


def test_volcano_paired_mode_excludes_invalid_pairs_before_calculation():
    numerator = pd.DataFrame(
        [[2.0, 0.0, 8.0, np.nan]],
        index=[100.0],
    )
    denominator = pd.DataFrame(
        [[1.0, 2.0, 2.0, 4.0]],
        index=[100.0],
    )

    fig, ax = plt.subplots()
    _, results = Figures.volcano(numerator, denominator, ax=ax, paired=True)

    assert np.isclose(results.loc[100.0, "fold_change"], 1.5)
    expected_pvalue = ttest_rel(np.log2([2.0, 8.0]), np.log2([1.0, 2.0])).pvalue
    assert np.isclose(results.loc[100.0, "pval"], expected_pvalue)
    plt.close(fig)


def test_volcano_accepts_a_custom_linear_fold_change_cutoff():
    denominator = pd.DataFrame(
        [[10.0, 11.0, 9.0, 10.5, 9.5]],
        index=[100.0],
    )
    numerator = denominator * 1.5

    default_fig, default_ax = plt.subplots()
    Figures.volcano(numerator, denominator, ax=default_ax)
    default_markers = [item for item in default_ax.collections if isinstance(item, PathCollection)]
    assert len(default_markers) == 1

    custom_fig, custom_ax = plt.subplots()
    Figures.volcano(
        numerator,
        denominator,
        ax=custom_ax,
        cutoff_specifier=1.25,
    )
    custom_markers = [item for item in custom_ax.collections if isinstance(item, PathCollection)]
    assert len(custom_markers) == 2
    vertical_cutoffs = sorted(
        float(line.get_xdata()[0])
        for line in custom_ax.lines
        if np.asarray(line.get_xdata()).size > 1
        and np.allclose(line.get_xdata(), line.get_xdata()[0])
    )
    np.testing.assert_allclose(vertical_cutoffs, [-np.log2(1.25), np.log2(1.25)])

    plt.close(default_fig)
    plt.close(custom_fig)


def test_volcano_rejects_invalid_fold_change_cutoff():
    numerator = pd.DataFrame([[2.0, 3.0]], index=[100.0])
    denominator = pd.DataFrame([[1.0, 1.5]], index=[100.0])

    with np.testing.assert_raises_regex(ValueError, "greater than 1"):
        Figures.volcano(numerator, denominator, cutoff_specifier=1.0)


def test_volcano_can_use_benjamini_hochberg_adjusted_pvalues(monkeypatch):
    raw_pvalues = np.array([0.01, 0.04, 0.2, 0.5])
    pvalue_iter = iter(raw_pvalues)
    monkeypatch.setattr(
        Figures,
        "ttest_ind",
        lambda *_args, **_kwargs: (0.0, next(pvalue_iter)),
    )
    denominator = pd.DataFrame(
        np.tile([1.0, 1.1, 0.9], (4, 1)),
        index=[100.0, 200.0, 300.0, 400.0],
    )
    numerator = denominator * 3.0

    fig, ax = plt.subplots()
    _, results = Figures.volcano(
        numerator,
        denominator,
        ax=ax,
        fdr_control=True,
    )

    expected_adjusted = false_discovery_control(raw_pvalues, method="bh")
    np.testing.assert_allclose(results["pval"], raw_pvalues)
    np.testing.assert_allclose(results["adjusted_pval"], expected_adjusted)
    base_points, significant_points = [
        item for item in ax.collections if isinstance(item, PathCollection)
    ]
    np.testing.assert_allclose(np.asarray(base_points.get_offsets())[:, 1], expected_adjusted)
    assert len(significant_points.get_offsets()) == 1
    assert ax.get_ylabel() == "BH-adjusted p-value"
    plt.close(fig)


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


def test_grouped_plots_allow_different_secondary_groups_and_key_types():
    data = {
        "First": {
            1: [1.0, 1.1, 1.2],
            "shared": [2.0, 2.1, 2.2],
        },
        "Second": {
            "shared": [3.0, 3.1, 3.2],
            "later only": [4.0, 4.1, 4.2],
        },
    }

    bar_fig, bar_ax = plt.subplots()
    Figures.barchart(data, ax=bar_ax, colors=["#2DB30C"])
    assert len(bar_ax.patches) == 4
    assert [text.get_text() for text in bar_ax.get_legend().get_texts()] == [
        "1", "shared", "later only"
    ]

    box_fig, box_ax = plt.subplots()
    assert Figures.boxplot(data, ax=box_ax, colors=["#2DB30C"]) is box_ax
    assert len(box_ax.patches) == 4
    assert [text.get_text() for text in box_ax.get_legend().get_texts()] == [
        "1", "shared", "later only"
    ]

    plt.close(bar_fig)
    plt.close(box_fig)


def test_violinplot_uses_the_boxplot_dataframe_workflow_with_missing_groups():
    data = pd.DataFrame({
        "sample": ["First"] * 6 + ["Second"] * 6,
        "condition": [1] * 3 + ["shared"] * 3 + ["shared"] * 3 + ["later only"] * 3,
        "signal": [1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2, 4.0, 4.1, 4.2],
    })

    fig, ax = plt.subplots()
    returned_ax = Figures.violinplot(
        data,
        ax=ax,
        value="signal",
        primary_group="sample",
        secondary_group="condition",
    )

    assert returned_ax is ax
    assert len(ax.collections) == 13
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["First", "Second"]
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "1", "shared", "later only"
    ]
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


def test_barchart_defaults_to_directional_standard_deviation_errorbars():
    data = {
        "Positive": [1, 2, 3],
        "Negative": [-1, -2, -3],
    }

    fig, ax = plt.subplots()
    Figures.barchart(data, ax=ax)

    error_cap_ys = sorted(
        float(value)
        for line in ax.lines
        for value in np.asarray(line.get_ydata(), dtype=float)
    )
    assert np.allclose(error_cap_ys, [-3, -2, 2, 3])
    plt.close(fig)


def test_barchart_can_disable_errors_for_grouped_and_ungrouped_data():
    datasets = [
        {"Control": [1, 2, 3], "Treatment": [2, 4, 6]},
        {
            "Sample": {
                "Control": [1, 2, 3],
                "Treatment": [2, 4, 6],
            }
        },
    ]

    for data in datasets:
        fig, ax = plt.subplots()
        Figures.barchart(data, ax=ax, draw_error=False)
        assert len(ax.lines) == 0
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
