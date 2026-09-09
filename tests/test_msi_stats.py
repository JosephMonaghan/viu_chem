from types import SimpleNamespace
import warnings

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
