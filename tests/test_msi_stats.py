from types import SimpleNamespace
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from viu_chem import msi_stats


def _msi_table(mz=(100.0, 200.0, 300.0)):
    matrix = sparse.csc_matrix(
        [
            [10.0, 1.0, 0.0],
            [9.0, 1.0, 0.0],
            [0.0, 1.0, 10.0],
            [0.0, 1.0, 9.0],
        ]
    )
    obs = pd.DataFrame(
        {
            "x": [0, 1, 0, 1],
            "y": [0, 0, 1, 1],
            "spatial_x": [0.0, 50.0, 0.0, 50.0],
            "spatial_y": [0.0, 0.0, 75.0, 75.0],
        }
    )
    var = pd.DataFrame({"mz": np.asarray(mz, dtype=float)})
    return SimpleNamespace(
        X=matrix,
        obs=obs,
        var=var,
        var_names=pd.Index([str(value) for value in mz]),
        n_obs=matrix.shape[0],
        n_vars=matrix.shape[1],
        uns={"coregistration_display_name": "nano-DESI (Positive)"},
    )


def test_kmeans_cluster_zarr_uses_named_sparse_table(monkeypatch):
    table = _msi_table()
    calls = []

    def load_table(path, dataset):
        calls.append((path, dataset))
        return table

    monkeypatch.setattr(msi_stats, "_load_zarr_msi_table", load_table)

    result = msi_stats.kmeans_cluster_zarr(
        "campaign.zarr",
        "nano-DESI (Positive)",
        n_clusters=2,
        min_cluster_size=1,
    )

    assert calls == [(msi_stats.Path("campaign.zarr"), "nano-DESI (Positive)")]
    assert result["sample"].unique().tolist() == ["nano-DESI (Positive)"]
    assert result["pixel_size_x"].unique().tolist() == [50.0]
    assert result["pixel_size_y"].unique().tolist() == [75.0]
    assert result.loc[0, "cluster"] == result.loc[1, "cluster"]
    assert result.loc[2, "cluster"] == result.loc[3, "cluster"]
    assert result.loc[0, "cluster"] != result.loc[2, "cluster"]
    assert result.attrs["tic_normalized"] is True
    assert result.attrs["source_format"] == "zarr"
    assert result.attrs["msi_dataset"] == "nano-DESI (Positive)"


def test_kmeans_cluster_zarr_rejects_different_self_aligned_axes(monkeypatch):
    tables = {
        "first.zarr": _msi_table((100.0, 200.0, 300.0)),
        "second.zarr": _msi_table((100.0, 200.1, 300.0)),
    }
    monkeypatch.setattr(
        msi_stats,
        "_load_zarr_msi_table",
        lambda path, _dataset: tables[path.name],
    )

    with pytest.raises(ValueError, match="common axis"):
        msi_stats.kmeans_cluster_zarr(
            ["first.zarr", "second.zarr"],
            "nano-DESI (Positive)",
            n_clusters=2,
        )


def test_kmeans_cluster_zarr_requires_pixel_coordinates(monkeypatch):
    table = _msi_table()
    table.obs = table.obs.drop(columns="x")
    monkeypatch.setattr(msi_stats, "_load_zarr_msi_table", lambda *_args: table)

    with pytest.raises(ValueError, match=r"obs\['x'\]"):
        msi_stats.kmeans_cluster_zarr(
            "campaign.zarr",
            "nano-DESI (Positive)",
            n_clusters=2,
        )


def test_mean_spectra_by_cluster_zarr_reuses_cluster_assignments(monkeypatch):
    table = _msi_table()
    monkeypatch.setattr(msi_stats, "_load_zarr_msi_table", lambda *_args: table)
    clusters = pd.DataFrame(
        {
            "sample": ["nano-DESI (Positive)"] * 4,
            "x": [0, 1, 0, 1],
            "y": [0, 0, 1, 1],
            "z": [1, 1, 1, 1],
            "cluster": [1, 1, 2, 2],
        }
    )
    clusters.attrs["tic_normalized"] = True

    mz_axis, means = msi_stats.mean_spectra_by_cluster_zarr(
        clusters,
        "campaign.zarr",
        "nano-DESI (Positive)",
    )

    np.testing.assert_array_equal(mz_axis, [100.0, 200.0, 300.0])
    np.testing.assert_allclose(means.sum(axis=0), [1.0, 1.0])
    assert means.loc[100.0, 1] > means.loc[300.0, 1]
    assert means.loc[300.0, 2] > means.loc[100.0, 2]
    assert means.attrs["source_format"] == "zarr"


def test_kmeans_rescales_large_raw_intensities_without_overflow():
    matrix = sparse.csr_matrix(
        [[1e200, 0.0], [9e199, 1e199], [0.0, 1e200], [1e199, 9e199]]
    )
    rows = [("sample", idx, 0, 1, 1.0, 1.0) for idx in range(4)]

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = msi_stats._cluster_spectral_matrix(
            matrix,
            rows,
            n_clusters=2,
            tic_normalize=False,
            random_state=42,
            n_init="auto",
            max_iter=300,
            auto_k_min=2,
            auto_k_max=10,
            min_cluster_fraction=0.01,
            min_cluster_size=1,
        )

    assert result.attrs["kmeans_input_scale"] == pytest.approx(1e200)


def test_umap_zarr_combines_aligned_campaign_without_mass_alignment(monkeypatch):
    tables = {"first.zarr": _msi_table(), "second.zarr": _msi_table()}
    captured = {}

    class FakeUMAP:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def fit_transform(self, matrix):
            captured["shape"] = matrix.shape
            captured["row_sums"] = np.asarray(matrix.sum(axis=1)).reshape(-1)
            return np.column_stack(
                [np.arange(matrix.shape[0]), -np.arange(matrix.shape[0])]
            )

    monkeypatch.setattr(
        msi_stats,
        "_load_zarr_msi_table",
        lambda path, _dataset: tables[path.name],
    )
    monkeypatch.setattr(msi_stats, "_load_umap_class", lambda: FakeUMAP)

    result = msi_stats.umap_zarr(
        ["first.zarr", "second.zarr"],
        "nano-DESI (Positive)",
        svd_components=None,
    )

    assert captured["shape"] == (8, 3)
    np.testing.assert_allclose(captured["row_sums"], np.ones(8))
    assert result.shape[0] == 8
    assert result["sample"].unique().tolist() == [
        "first: nano-DESI (Positive)",
        "second: nano-DESI (Positive)",
    ]
    assert {"pixel_id", "x", "y", "UMAP_1", "UMAP_2"}.issubset(result.columns)
    assert result.attrs["mz_min"] == 100.0
    assert result.attrs["mz_max"] == 300.0
    assert result.attrs["n_mz_features"] == 3
    assert "UMAP_1" in repr(result)
    figure, axis = msi_stats.plot_umap(
        result,
        color_by="sample",
        legend_outside=True,
    )
    assert axis.get_xlabel() == "UMAP 1"
    assert figure.subplotpars.right == pytest.approx(0.72)
    plt.close(figure)


def test_umap_zarr_rejects_unaligned_campaign(monkeypatch):
    tables = {
        "first.zarr": _msi_table((100.0, 200.0, 300.0)),
        "second.zarr": _msi_table((100.0, 200.1, 300.0)),
    }
    monkeypatch.setattr(
        msi_stats,
        "_load_zarr_msi_table",
        lambda path, _dataset: tables[path.name],
    )

    with pytest.raises(ValueError, match="No mass-axis alignment"):
        msi_stats.umap_zarr(
            ["first.zarr", "second.zarr"],
            "nano-DESI (Positive)",
        )


def test_umap_zarr_applies_inclusive_mz_feature_range(monkeypatch):
    table = _msi_table()
    captured = {}

    class FakeUMAP:
        def __init__(self, **_kwargs):
            pass

        def fit_transform(self, matrix):
            captured["shape"] = matrix.shape
            return np.zeros((matrix.shape[0], 2), dtype=float)

    monkeypatch.setattr(msi_stats, "_load_zarr_msi_table", lambda *_args: table)
    monkeypatch.setattr(msi_stats, "_load_umap_class", lambda: FakeUMAP)

    result = msi_stats.umap_zarr(
        "campaign.zarr",
        "nano-DESI (Positive)",
        mz_range=(100.0, 200.0),
        svd_components=None,
    )

    assert captured["shape"] == (4, 2)
    assert result.attrs["mz_range_requested"] == (100.0, 200.0)
    assert result.attrs["mz_min"] == 100.0
    assert result.attrs["mz_max"] == 200.0
    assert result.attrs["n_mz_features"] == 2
    assert result.attrs["n_source_mz_features"] == 3


def test_umap_zarr_rejects_empty_mz_feature_range(monkeypatch):
    monkeypatch.setattr(msi_stats, "_load_zarr_msi_table", lambda *_args: _msi_table())

    with pytest.raises(ValueError, match="No m/z features"):
        msi_stats.umap_zarr(
            "campaign.zarr",
            "nano-DESI (Positive)",
            mz_range=(400.0, 500.0),
        )
