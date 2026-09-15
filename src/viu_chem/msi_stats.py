import warnings
from collections.abc import Sequence
from pyimzml import ImzMLParser
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import pairwise_distances_argmin
import matplotlib.pyplot as plt
import matplotlib as mpl
import re
from scipy.signal import find_peaks


def _normalize_imzml_paths(imzml_paths: str | Path | Sequence[str | Path]) -> list[Path]:
    """Normalizes one or more imzML path inputs into a non-empty list of Paths.
    
    :param imzml_paths: Single imzML path or sequence of imzML paths
    :return: List of imzML paths as Path objects"""
    if isinstance(imzml_paths, (str, Path)):
        return [Path(imzml_paths)]
    if not isinstance(imzml_paths, Sequence) or len(imzml_paths) == 0:
        raise ValueError("imzml_paths must be a path or a non-empty sequence of paths.")
    return [Path(path) for path in imzml_paths]


def _normalize_zarr_paths(zarr_paths: str | Path | Sequence[str | Path]) -> list[Path]:
    """Normalize one or more SpatialData Zarr paths into a non-empty list."""
    if isinstance(zarr_paths, (str, Path)):
        return [Path(zarr_paths)]
    if not isinstance(zarr_paths, Sequence) or len(zarr_paths) == 0:
        raise ValueError("zarr_paths must be a path or a non-empty sequence of paths.")
    return [Path(path) for path in zarr_paths]


def _validate_kmeans_options(
    n_clusters: int | str,
    min_cluster_fraction: float,
    min_cluster_size: int,
) -> None:
    if isinstance(n_clusters, str) and n_clusters != "auto":
        raise ValueError("n_clusters must be an integer >= 1 or 'auto'.")
    if not isinstance(n_clusters, (int, str)) or isinstance(n_clusters, bool):
        raise ValueError("n_clusters must be an integer >= 1 or 'auto'.")
    if isinstance(n_clusters, int) and n_clusters < 1:
        raise ValueError("n_clusters must be at least 1.")
    if min_cluster_fraction < 0:
        raise ValueError("min_cluster_fraction must be >= 0.")
    if min_cluster_size < 1:
        raise ValueError("min_cluster_size must be >= 1.")


def _cluster_spectral_matrix(
    data,
    row_info: list[tuple],
    *,
    n_clusters: int | str,
    tic_normalize: bool,
    random_state: int | None,
    n_init: int | str,
    max_iter: int,
    auto_k_min: int,
    auto_k_max: int,
    min_cluster_fraction: float,
    min_cluster_size: int,
) -> pd.DataFrame:
    """Run the clustering shared by the imzML and Zarr loaders."""
    _validate_kmeans_options(n_clusters, min_cluster_fraction, min_cluster_size)
    n_pixels = data.shape[0]
    if n_pixels < 1:
        raise ValueError("No spectra found to cluster.")
    if len(row_info) != n_pixels:
        raise ValueError("Spectrum metadata does not match the number of spectra.")

    if sparse.issparse(data):
        data = data.tocsr().astype(float, copy=False)
    else:
        data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("Spectral data must be a two-dimensional pixels x m/z matrix.")
    values = data.data if sparse.issparse(data) else data
    if not np.all(np.isfinite(values)):
        raise ValueError("Spectral data contains NaN or infinite intensity values.")

    # K-means is invariant to a uniform positive scale factor. Bounding the
    # largest value prevents squared distances in k-means++ from overflowing.
    max_abs_intensity = float(np.max(np.abs(values))) if values.size else 0.0
    kmeans_input_scale = max(1.0, max_abs_intensity)
    if kmeans_input_scale > 1.0:
        data = data / kmeans_input_scale
    if tic_normalize:
        data = _tic_normalize_matrix(data)

    auto_mode = n_clusters == "auto"
    if auto_mode:
        auto_k = int(round(np.sqrt(n_pixels)))
        initial_k = min(max(auto_k_min, auto_k), auto_k_max, n_pixels)
    else:
        initial_k = min(int(n_clusters), n_pixels)

    model = KMeans(
        n_clusters=initial_k,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
    )
    labels0 = model.fit_predict(data)

    # In auto mode, reassign tiny clusters to the nearest retained centroid.
    # pairwise_distances_argmin supports both dense and sparse input matrices.
    if auto_mode:
        counts = np.bincount(labels0, minlength=initial_k)
        min_size_threshold = max(
            min_cluster_size,
            int(np.ceil(min_cluster_fraction * n_pixels)),
        )
        keep = np.where(counts >= min_size_threshold)[0]
        if keep.size == 0:
            keep = np.array([int(np.argmax(counts))], dtype=int)

        drop = np.setdiff1d(np.arange(initial_k), keep, assume_unique=True)
        labels_adj = labels0.copy()
        if drop.size > 0:
            dropped_mask = np.isin(labels_adj, drop)
            if np.any(dropped_mask):
                nearest_keep_idx = pairwise_distances_argmin(
                    data[dropped_mask],
                    model.cluster_centers_[keep],
                    metric="euclidean",
                )
                labels_adj[dropped_mask] = keep[nearest_keep_idx]
        unique_labels = np.sort(np.unique(labels_adj))
        remap = {old: new for new, old in enumerate(unique_labels, start=1)}
        labels = np.array([remap[val] for val in labels_adj], dtype=int)
        final_k = len(unique_labels)
    else:
        labels = labels0 + 1
        final_k = initial_k

    df = pd.DataFrame(
        row_info,
        columns=["sample", "x", "y", "z", "pixel_size_x", "pixel_size_y"],
    )
    df["cluster"] = labels
    df.attrs["tic_normalized"] = bool(tic_normalize)
    df.attrs["k_requested"] = n_clusters
    df.attrs["k_initial"] = int(initial_k)
    df.attrs["k_final"] = int(final_k)
    df.attrs["kmeans_input_scale"] = kmeans_input_scale
    if auto_mode:
        df.attrs["min_cluster_fraction"] = float(min_cluster_fraction)
        df.attrs["min_cluster_size"] = int(min_cluster_size)
        df.attrs["min_size_threshold_used"] = int(min_size_threshold)
    return df


def _validate_file_continuous(imzml) -> None:
    """Validates that an imzML parser represents a continuous aligned file.
    
    :param imzml: imzML parser object to validate"""
    metadata = imzml.metadata.pretty()
    is_continuous = metadata["file_description"]["continuous"]
    if not is_continuous:
        raise TypeError("imzML file must be continuous (aligned m/z).")


def _to_float_or_none(value) -> float | None:
    """Extracts a float from a numeric value or numeric string when possible.
    
    :param value: Value to convert
    :return: Float value, or None if no numeric value can be parsed"""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
        if match:
            return float(match.group(0))
    return None


def _collect_numeric_metadata(metadata_obj) -> dict[str, float]:
    """Collects numeric metadata values from a nested metadata object.
    
    :param metadata_obj: Metadata object containing nested dictionaries or sequences
    :return: Dictionary mapping normalized metadata keys to numeric values"""
    out: dict[str, float] = {}

    def _walk(obj):
        """Recursively walks nested metadata values and stores first numeric matches.
        
        :param obj: Metadata object or nested value to inspect"""
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_norm = str(key).strip().lower()
                num_val = _to_float_or_none(value)
                if num_val is not None and key_norm not in out:
                    out[key_norm] = num_val
                _walk(value)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                _walk(value)

    _walk(metadata_obj)
    return out


def _get_pixel_size(imzml) -> tuple[float, float]:
    """Determines x and y pixel sizes from imzML metadata.
    
    :param imzml: imzML parser object
    :return: Tuple of x and y pixel sizes"""
    metadata = imzml.metadata.pretty()
    numeric_meta = _collect_numeric_metadata(metadata)
    x_count = float(imzml.imzmldict.get("max count of pixels x", 0)) + 1.0
    y_count = float(imzml.imzmldict.get("max count of pixels y", 0)) + 1.0

    x_keys = ("pixel size (x)", "pixel size x")
    y_keys = ("pixel size (y)", "pixel size y")

    pixel_size_x = None
    pixel_size_y = None

    for key in x_keys:
        if key in numeric_meta:
            pixel_size_x = numeric_meta[key]
            break
    for key in y_keys:
        if key in numeric_meta:
            pixel_size_y = numeric_meta[key]
            break

    if pixel_size_x is None and "max dimension x" in numeric_meta and x_count > 0:
        pixel_size_x = numeric_meta["max dimension x"] / x_count
    if pixel_size_y is None and "max dimension y" in numeric_meta and y_count > 0:
        pixel_size_y = numeric_meta["max dimension y"] / y_count

    if pixel_size_x is None:
        pixel_size_x = 1.0
    if pixel_size_y is None:
        pixel_size_y = 1.0

    return pixel_size_x, pixel_size_y


def _cluster_color_mapping(
    cluster_labels: Sequence[int] | np.ndarray,
    cmap: str = "tab20",
) -> dict[int, tuple[float, float, float, float]]:
    """Maps cluster labels to RGBA colors from a matplotlib colormap.
    
    :param cluster_labels: Cluster labels to map
    :param cmap: Matplotlib colormap name
    :return: Dictionary mapping cluster labels to RGBA colors"""
    labels = np.sort(np.asarray(cluster_labels, dtype=int))
    cmap_obj = plt.get_cmap(cmap, len(labels))
    return {int(label): cmap_obj(idx) for idx, label in enumerate(labels)}


def _prominent_peak_indices(
    mz_axis: np.ndarray,
    intensities: np.ndarray,
    max_labels: int = 8,
    min_rel_prominence: float = 0.05,
) -> np.ndarray:
    """Finds prominent peak indices for spectrum labeling.
    
    :param mz_axis: m/z axis values
    :param intensities: Spectrum intensity values
    :param max_labels: Maximum number of peaks to label
    :param min_rel_prominence: Minimum relative prominence required for labeling
    :return: Array of selected peak indices"""
    if max_labels < 1:
        return np.array([], dtype=int)
    y = np.asarray(intensities, dtype=float)
    if y.size == 0:
        return np.array([], dtype=int)
    y_span = float(np.nanmax(y) - np.nanmin(y))
    if y_span <= 0:
        return np.array([], dtype=int)

    # Use a mild distance constraint to reduce local over-labeling.
    min_distance = max(1, y.size // 250)
    peak_idx, props = find_peaks(
        y,
        prominence=max(min_rel_prominence * y_span, 0.0),
        distance=min_distance,
    )
    if peak_idx.size == 0:
        return peak_idx

    prominences = props.get("prominences", np.zeros_like(peak_idx, dtype=float))
    order = np.argsort(prominences)[::-1]
    selected = peak_idx[order[:max_labels]]
    selected.sort()
    return selected


def _annotate_peaks(
    ax,
    mz_axis: np.ndarray,
    intensities: np.ndarray,
    color,
    max_labels: int = 8,
    min_rel_prominence: float = 0.05,
):
    """Annotates prominent peaks on a spectrum axis.
    
    :param ax: Matplotlib axes object to annotate
    :param mz_axis: m/z axis values
    :param intensities: Spectrum intensity values
    :param color: Annotation text color
    :param max_labels: Maximum number of peaks to label
    :param min_rel_prominence: Minimum relative prominence required for labeling"""
    peak_idx = _prominent_peak_indices(
        mz_axis=mz_axis,
        intensities=intensities,
        max_labels=max_labels,
        min_rel_prominence=min_rel_prominence,
    )
    if peak_idx.size == 0:
        return

    for idx in peak_idx:
        mz_val = float(mz_axis[idx])
        y_val = float(intensities[idx])
        ax.annotate(
            f"{mz_val:.4f}",
            xy=(mz_val, y_val),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color=color,
        )

def get_mean_spectrum(img:Path,normalize:bool=False):
    """Extracts the mean spectrum from an aligned imzML file.
    
    :param img: Path to a continuous aligned imzML file
    :param normalize: Whether or not to TIC normalize each spectrum before averaging
    :return: Tuple of m/z axis and average intensity values"""

    with warnings.catch_warnings(action='ignore'):
        imzml = ImzMLParser.ImzMLParser(img)

    _validate_file_continuous(imzml)
    
    mz, intensity = imzml.getspectrum(0)
    if normalize:
        intensity = intensity / intensity.sum()
    for idx, coord in enumerate(imzml.coordinates):
        if idx == 0:
            continue
        _, local_int = imzml.getspectrum(idx)
        if normalize:
            if local_int.sum() > 0:
                local_int = local_int / local_int.sum()
        intensity = local_int + intensity
    
    average_int = intensity / len(imzml.coordinates)
    return mz, average_int


def kmeans_cluster_imzml(
    imzml_paths: str | Path | Sequence[str | Path],
    n_clusters: int | str,
    tic_normalize: bool = True,
    random_state: int | None = 42,
    n_init: int | str = "auto",
    max_iter: int = 300,
    auto_k_min: int = 2,
    auto_k_max: int = 10,
    min_cluster_fraction: float = 0.01,
    min_cluster_size: int = 25,
) -> pd.DataFrame:
    """Runs k-means clustering on one or more continuous aligned imzML datasets.
    
    :param imzml_paths: One imzML path or a sequence of imzML paths
    :param n_clusters: Number of clusters to compute, or "auto"
    :param tic_normalize: Whether to TIC-normalize spectra before clustering
    :param random_state: Random state passed to sklearn KMeans
    :param n_init: Number of initializations for KMeans
    :param max_iter: Maximum number of k-means iterations
    :param auto_k_min: Minimum initial k when n_clusters is "auto"
    :param auto_k_max: Maximum initial k when n_clusters is "auto"
    :param min_cluster_fraction: Minimum fraction of total pixels a cluster must contain
    :param min_cluster_size: Minimum absolute pixel count a cluster must contain
    :return: Dataframe containing sample, coordinates, pixel sizes, and cluster labels"""
    paths = _normalize_imzml_paths(imzml_paths)
    spectra_blocks: list[np.ndarray] = []
    row_info: list[tuple[str, int, int, int, float, float]] = []
    reference_mz: np.ndarray | None = None

    for path in paths:
        with warnings.catch_warnings(action="ignore"):
            imzml = ImzMLParser.ImzMLParser(path)

        _validate_file_continuous(imzml)
        pixel_size_x, pixel_size_y = _get_pixel_size(imzml)
        local_mz, local_intensity = imzml.getspectrum(0)
        local_mz = np.asarray(local_mz)
        local_intensity = np.asarray(local_intensity, dtype=float)
        local_spectra = [local_intensity]
        x0, y0, z0 = imzml.coordinates[0]
        row_info.append((path.stem, x0, y0, z0, pixel_size_x, pixel_size_y))

        for idx, coord in enumerate(imzml.coordinates):
            if idx == 0:
                continue
            mz, intensity = imzml.getspectrum(idx)
            mz = np.asarray(mz)
            if not np.array_equal(mz, local_mz):
                raise TypeError(f"imzML file must be continuous (aligned m/z): {path}")
            intensity = np.asarray(intensity, dtype=float)
            local_spectra.append(intensity)
            x, y, z = coord
            row_info.append((path.stem, x, y, z, pixel_size_x, pixel_size_y))

        if reference_mz is None:
            reference_mz = local_mz
        elif not np.array_equal(local_mz, reference_mz):
            raise ValueError(
                "All input imzML files must share the same m/z axis for joint clustering."
            )

        spectra_blocks.append(np.vstack(local_spectra))

    return _cluster_spectral_matrix(
        np.vstack(spectra_blocks),
        row_info,
        n_clusters=n_clusters,
        tic_normalize=tic_normalize,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        auto_k_min=auto_k_min,
        auto_k_max=auto_k_max,
        min_cluster_fraction=min_cluster_fraction,
        min_cluster_size=min_cluster_size,
    )


def _load_zarr_msi_table(zarr_path: Path, msi_dataset: str):
    """Load one selected table without requiring coregistration dependencies at import time."""
    try:
        from viu_chem.msi_coregistration import get_msi_table
    except ImportError as exc:
        raise ImportError(
            "Zarr clustering requires the viu-chem coregistration dependencies."
        ) from exc
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The table is annotating .*not present in the SpatialData object\.",
            category=UserWarning,
        )
        return get_msi_table(zarr_path, msi_dataset)


def _table_mz_axis(table) -> np.ndarray:
    if "mz" in table.var:
        mz_axis = np.asarray(table.var["mz"], dtype=float)
    else:
        try:
            mz_axis = np.asarray(table.var_names, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "The selected MSI table must provide its mass axis in var['mz'] "
                "or as numeric var_names."
            ) from exc
    if mz_axis.ndim != 1 or mz_axis.size != table.n_vars:
        raise ValueError("The MSI table m/z axis does not match its spectral matrix.")
    return mz_axis


def _axis_scale_from_obs(obs: pd.DataFrame, axis: str) -> float | None:
    explicit_key = f"pixel_size_{axis}"
    if explicit_key in obs:
        values = pd.to_numeric(obs[explicit_key], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        if values.size:
            return float(np.median(values))

    spatial_key = f"spatial_{axis}"
    if axis not in obs or spatial_key not in obs:
        return None
    coords = pd.DataFrame(
        {
            "grid": pd.to_numeric(obs[axis], errors="coerce"),
            "spatial": pd.to_numeric(obs[spatial_key], errors="coerce"),
        }
    ).dropna()
    if coords.empty:
        return None
    coords = coords.groupby("grid", as_index=False)["spatial"].median().sort_values("grid")
    grid_diff = np.diff(coords["grid"].to_numpy(dtype=float))
    spatial_diff = np.diff(coords["spatial"].to_numpy(dtype=float))
    valid = np.isfinite(grid_diff) & np.isfinite(spatial_diff) & (grid_diff != 0)
    scales = np.abs(spatial_diff[valid] / grid_diff[valid])
    scales = scales[np.isfinite(scales) & (scales > 0)]
    return float(np.median(scales)) if scales.size else None


def _table_pixel_sizes(table) -> tuple[float, float]:
    uns = getattr(table, "uns", {}) or {}

    def _size(axis: str) -> float:
        for key in (f"pixel_size_{axis}_um", f"pixel_size_{axis}"):
            value = _to_float_or_none(uns.get(key))
            if value is not None and value > 0:
                return value
        return _axis_scale_from_obs(table.obs, axis) or 1.0

    return _size("x"), _size("y")


def _table_spectral_matrix(table):
    matrix = table.X
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
    elif hasattr(matrix, "compute"):
        matrix = matrix.compute()
    if sparse.issparse(matrix):
        return matrix.tocsr().astype(float, copy=False)
    return np.asarray(matrix, dtype=float)


def _zarr_sample_name(path: Path, table, n_paths: int, msi_dataset: str) -> str:
    display_name = str(
        table.uns.get("coregistration_display_name")
        or table.uns.get("coregistration_dataset_label")
        or msi_dataset
    )
    return display_name if n_paths == 1 else f"{path.stem}: {display_name}"


def _load_umap_class():
    try:
        from umap import UMAP
    except ImportError as exc:
        import sys

        loaded_module = sys.modules.get("umap")
        resolved_path = getattr(loaded_module, "__file__", None)
        location_hint = f" Module resolved to: {resolved_path}." if resolved_path else ""
        raise ImportError(
            "Could not import UMAP from umap-learn. "
            f"Python interpreter: {sys.executable}.{location_hint} "
            f"Original import error: {exc}. "
            "Ensure umap-learn is installed in this interpreter and that the "
            "script/current directory does not contain a file named `umap.py`."
        ) from exc
    return UMAP


def _svd_for_umap(data, n_components: int, random_state: int | None):
    """Reduce UMAP input and retry with ARPACK if randomized SVD is non-finite."""
    randomized = TruncatedSVD(
        n_components=n_components,
        random_state=random_state,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        reduced = randomized.fit_transform(data)
    runtime_warnings = [str(item.message) for item in caught]
    if np.all(np.isfinite(reduced)):
        return reduced, "randomized", runtime_warnings

    arpack = TruncatedSVD(
        n_components=n_components,
        algorithm="arpack",
        random_state=random_state,
    )
    reduced = arpack.fit_transform(data)
    if not np.all(np.isfinite(reduced)):
        raise ValueError(
            "Both randomized and ARPACK truncated SVD produced non-finite values."
        )
    return reduced, "arpack", runtime_warnings


def _tic_normalize_matrix(data):
    """TIC-normalize dense or sparse rows while protecting sums from overflow."""
    values = data.data if sparse.issparse(data) else data
    if not np.all(np.isfinite(values)):
        raise ValueError("Spectral data contains NaN or infinite intensity values.")
    max_abs_intensity = float(np.max(np.abs(values))) if values.size else 0.0
    if max_abs_intensity > 1.0:
        data = data / max_abs_intensity
    row_sums = np.asarray(data.sum(axis=1)).reshape(-1)
    inverse_tic = np.zeros_like(row_sums, dtype=float)
    positive = row_sums > 0
    inverse_tic[positive] = 1.0 / row_sums[positive]
    if sparse.issparse(data):
        return data.multiply(inverse_tic[:, None]).tocsr()
    return data * inverse_tic[:, None]


def kmeans_cluster_zarr(
    zarr_paths: str | Path | Sequence[str | Path],
    msi_dataset: str,
    n_clusters: int | str,
    tic_normalize: bool = True,
    random_state: int | None = 42,
    n_init: int | str = "auto",
    max_iter: int = 300,
    auto_k_min: int = 2,
    auto_k_max: int = 10,
    min_cluster_fraction: float = 0.01,
    min_cluster_size: int = 25,
) -> pd.DataFrame:
    """Run k-means on an MSI table selected from one or more SpatialData Zarr stores.

    The selected table must use the standard ``pixels x m/z`` AnnData layout,
    with coordinates in ``obs['x']`` and ``obs['y']`` and the mass axis in
    ``var['mz']``. Sparse spectral matrices remain sparse during stacking,
    TIC normalization, and k-means fitting.

    :param zarr_paths: One SpatialData Zarr path or a sequence of paths
    :param msi_dataset: Display name, label, table key, or TIC key of the MSI dataset
    :param n_clusters: Number of clusters to compute, or ``"auto"``
    :param tic_normalize: Whether to TIC-normalize spectra before clustering
    :param random_state: Random state passed to sklearn KMeans
    :param n_init: Number of initializations for KMeans
    :param max_iter: Maximum number of k-means iterations
    :param auto_k_min: Minimum initial k when n_clusters is ``"auto"``
    :param auto_k_max: Maximum initial k when n_clusters is ``"auto"``
    :param min_cluster_fraction: Minimum fraction of pixels retained as a cluster
    :param min_cluster_size: Minimum absolute pixel count retained as a cluster
    :return: Dataframe containing sample, coordinates, pixel sizes, and cluster labels
    """
    _validate_kmeans_options(n_clusters, min_cluster_fraction, min_cluster_size)
    paths = _normalize_zarr_paths(zarr_paths)
    spectra_blocks = []
    row_info: list[tuple] = []
    reference_mz: np.ndarray | None = None

    for path in paths:
        table = _load_zarr_msi_table(path, msi_dataset)
        if "x" not in table.obs or "y" not in table.obs:
            raise ValueError(
                f"MSI table {msi_dataset!r} in {path} must contain obs['x'] and obs['y']."
            )

        local_mz = _table_mz_axis(table)
        if reference_mz is None:
            reference_mz = local_mz
        elif not np.array_equal(local_mz, reference_mz):
            raise ValueError(
                "All selected Zarr MSI tables must share the same m/z axis for "
                "joint clustering. Run self-aligned datasets separately or align "
                "them to a common axis first."
            )

        matrix = _table_spectral_matrix(table)
        if matrix.shape != (table.n_obs, table.n_vars):
            raise ValueError(f"Unexpected spectral matrix shape in {path}: {matrix.shape}")
        spectra_blocks.append(matrix)

        pixel_size_x, pixel_size_y = _table_pixel_sizes(table)
        display_name = str(
            table.uns.get("coregistration_display_name")
            or table.uns.get("coregistration_dataset_label")
            or msi_dataset
        )
        sample_name = display_name if len(paths) == 1 else f"{path.stem}: {display_name}"
        x_values = table.obs["x"].to_numpy()
        y_values = table.obs["y"].to_numpy()
        z_values = table.obs["z"].to_numpy() if "z" in table.obs else np.ones(table.n_obs, dtype=int)
        row_info.extend(
            (sample_name, x, y, z, pixel_size_x, pixel_size_y)
            for x, y, z in zip(x_values, y_values, z_values, strict=True)
        )

    if any(sparse.issparse(block) for block in spectra_blocks):
        data = sparse.vstack(
            [block if sparse.issparse(block) else sparse.csr_matrix(block) for block in spectra_blocks],
            format="csr",
        )
    else:
        data = np.vstack(spectra_blocks)

    result = _cluster_spectral_matrix(
        data,
        row_info,
        n_clusters=n_clusters,
        tic_normalize=tic_normalize,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        auto_k_min=auto_k_min,
        auto_k_max=auto_k_max,
        min_cluster_fraction=min_cluster_fraction,
        min_cluster_size=min_cluster_size,
    )
    result.attrs["source_format"] = "zarr"
    result.attrs["msi_dataset"] = str(msi_dataset)
    return result


def umap_zarr(
    zarr_paths: str | Path | Sequence[str | Path],
    msi_dataset: str,
    *,
    mz_range: tuple[float, float] | None = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    metric: str = "cosine",
    tic_normalize: bool = True,
    log_transform: bool = False,
    svd_components: int | None = 50,
    random_state: int | None = 42,
) -> pd.DataFrame:
    """Compute a pixel-level UMAP for one aligned Zarr or a campaign of Zarrs.

    Campaign tables are required to have exactly identical m/z axes. This
    function never bins, interpolates, or otherwise aligns mass features.
    Sparse matrices remain sparse through loading and TIC normalization. By
    default, truncated SVD reduces the aligned feature matrix before UMAP.

    :param zarr_paths: One SpatialData Zarr path or a sequence of campaign paths
    :param msi_dataset: Display name, label, table key, or TIC key to select
    :param mz_range: Optional inclusive ``(minimum, maximum)`` m/z feature range
    :param n_neighbors: UMAP local-neighborhood size
    :param min_dist: UMAP minimum embedding distance
    :param n_components: Number of UMAP dimensions
    :param metric: Distance metric passed to UMAP
    :param tic_normalize: Whether to TIC-normalize each pixel spectrum
    :param log_transform: Whether to apply ``log1p`` after TIC normalization
    :param svd_components: Truncated-SVD dimensions before UMAP, or ``None``
        to run UMAP directly on the aligned spectra
    :param random_state: Random seed for SVD and UMAP
    :return: Dataframe containing UMAP coordinates and source ``obs`` metadata
    """
    paths = _normalize_zarr_paths(zarr_paths)
    if mz_range is not None:
        if len(mz_range) != 2:
            raise ValueError("mz_range must contain exactly (minimum, maximum).")
        mz_min, mz_max = float(mz_range[0]), float(mz_range[1])
        if not np.all(np.isfinite([mz_min, mz_max])) or mz_min >= mz_max:
            raise ValueError("mz_range must contain two finite increasing values.")
    else:
        mz_min = mz_max = None
    if not isinstance(n_neighbors, int) or isinstance(n_neighbors, bool) or n_neighbors < 2:
        raise ValueError("n_neighbors must be an integer of at least 2.")
    if not isinstance(n_components, int) or isinstance(n_components, bool) or n_components < 1:
        raise ValueError("n_components must be an integer of at least 1.")
    if not np.isfinite(min_dist) or min_dist < 0:
        raise ValueError("min_dist must be a finite value greater than or equal to zero.")
    if svd_components is not None and (
        not isinstance(svd_components, int)
        or isinstance(svd_components, bool)
        or svd_components < 1
    ):
        raise ValueError("svd_components must be a positive integer or None.")

    spectra_blocks = []
    metadata_blocks: list[pd.DataFrame] = []
    reference_mz: np.ndarray | None = None
    selected_mz: np.ndarray | None = None
    mz_mask: np.ndarray | None = None
    for path in paths:
        table = _load_zarr_msi_table(path, msi_dataset)
        local_mz = _table_mz_axis(table)
        if not np.all(np.isfinite(local_mz)):
            raise ValueError(f"The MSI m/z axis in {path} contains non-finite values.")
        if reference_mz is None:
            reference_mz = local_mz
            if mz_range is None:
                mz_mask = np.ones(reference_mz.size, dtype=bool)
            else:
                mz_mask = (reference_mz >= mz_min) & (reference_mz <= mz_max)
            if not np.any(mz_mask):
                raise ValueError(
                    f"No m/z features fall inside the requested range {mz_range}."
                )
            selected_mz = reference_mz[mz_mask]
        elif not np.array_equal(local_mz, reference_mz):
            raise ValueError(
                "All campaign Zarr MSI tables must already share exactly the same "
                "m/z axis. No mass-axis alignment is performed by umap_zarr."
            )

        matrix = _table_spectral_matrix(table)
        if matrix.shape != (table.n_obs, table.n_vars):
            raise ValueError(f"Unexpected spectral matrix shape in {path}: {matrix.shape}")
        spectra_blocks.append(matrix[:, mz_mask])

        metadata = table.obs.reset_index(drop=True).copy()
        for reserved in ("pixel_id", "sample", "source_path"):
            if reserved in metadata:
                metadata.rename(columns={reserved: f"obs_{reserved}"}, inplace=True)
        metadata.insert(0, "pixel_id", table.obs.index.astype(str).to_numpy())
        metadata.insert(
            1,
            "sample",
            _zarr_sample_name(path, table, len(paths), msi_dataset),
        )
        metadata.insert(2, "source_path", str(path))
        metadata_blocks.append(metadata)

    if any(sparse.issparse(block) for block in spectra_blocks):
        data = sparse.vstack(
            [block if sparse.issparse(block) else sparse.csr_matrix(block) for block in spectra_blocks],
            format="csr",
        )
    else:
        data = np.vstack(spectra_blocks)
    if data.shape[0] < 3:
        raise ValueError("UMAP analysis requires at least three pixel spectra.")

    values = data.data if sparse.issparse(data) else data
    if not np.all(np.isfinite(values)):
        raise ValueError("Spectral data contains NaN or infinite intensity values.")
    max_abs_intensity = float(np.max(np.abs(values))) if values.size else 0.0
    input_scale = max(1.0, max_abs_intensity)
    if input_scale > 1.0:
        data = data / input_scale
    if tic_normalize:
        data = _tic_normalize_matrix(data)
    if log_transform:
        values = data.data if sparse.issparse(data) else data
        if np.any(values < 0):
            raise ValueError("log_transform requires non-negative intensity values.")
        if sparse.issparse(data):
            data = data.copy()
            data.data = np.log1p(data.data)
        else:
            data = np.log1p(data)

    effective_svd_components = None
    svd_algorithm = None
    svd_runtime_warnings: list[str] = []
    if svd_components is not None:
        effective_svd_components = min(
            svd_components,
            data.shape[0] - 1,
            data.shape[1] - 1,
        )
        if effective_svd_components >= 1:
            umap_input, svd_algorithm, svd_runtime_warnings = _svd_for_umap(
                data,
                effective_svd_components,
                random_state,
            )
        else:
            umap_input = data
            effective_svd_components = None
    else:
        umap_input = data

    effective_neighbors = min(n_neighbors, data.shape[0] - 1)
    reducer = _load_umap_class()(
        n_neighbors=effective_neighbors,
        min_dist=float(min_dist),
        n_components=n_components,
        metric=metric,
        random_state=random_state,
        n_jobs=1 if random_state is not None else -1,
    )
    embedding = np.asarray(reducer.fit_transform(umap_input), dtype=float)
    if embedding.shape != (data.shape[0], n_components):
        raise ValueError(f"UMAP returned an unexpected embedding shape: {embedding.shape}")
    if not np.all(np.isfinite(embedding)):
        raise ValueError("UMAP returned NaN or infinite embedding coordinates.")

    result = pd.concat(metadata_blocks, ignore_index=True)
    for index in range(n_components):
        result[f"UMAP_{index + 1}"] = embedding[:, index]
    result.attrs["source_format"] = "zarr"
    result.attrs["msi_dataset"] = str(msi_dataset)
    result.attrs["tic_normalized"] = bool(tic_normalize)
    result.attrs["log_transformed"] = bool(log_transform)
    result.attrs["metric"] = str(metric)
    result.attrs["n_neighbors"] = int(effective_neighbors)
    result.attrs["min_dist"] = float(min_dist)
    result.attrs["n_components"] = int(n_components)
    result.attrs["svd_components"] = effective_svd_components
    result.attrs["svd_algorithm"] = svd_algorithm
    result.attrs["svd_runtime_warnings"] = svd_runtime_warnings
    result.attrs["input_scale"] = input_scale
    result.attrs["mz_range_requested"] = mz_range
    result.attrs["mz_min"] = float(selected_mz[0])
    result.attrs["mz_max"] = float(selected_mz[-1])
    result.attrs["n_mz_features"] = int(selected_mz.size)
    result.attrs["source_mz_min"] = float(reference_mz[0])
    result.attrs["source_mz_max"] = float(reference_mz[-1])
    result.attrs["n_source_mz_features"] = int(reference_mz.size)
    return result


def plot_umap(
    umap_df: pd.DataFrame,
    color_by: str | None = "sample",
    *,
    ax=None,
    cmap: str = "viridis",
    dot_size: float = 4.0,
    alpha: float = 0.8,
    legend_outside: bool = False,
):
    """Plot the first two dimensions returned by :func:`umap_zarr`.

    :param umap_df: Dataframe returned by :func:`umap_zarr`
    :param color_by: Numeric or categorical column used to color pixels
    :param ax: Optional matplotlib axis
    :param cmap: Matplotlib colormap name
    :param dot_size: Scatter-point size
    :param alpha: Scatter-point opacity
    :param legend_outside: Place a categorical legend in a right-side margin
    :return: Matplotlib figure and axis
    """
    required = {"UMAP_1", "UMAP_2"}
    if not required.issubset(umap_df):
        raise ValueError("umap_df must contain UMAP_1 and UMAP_2 columns.")
    if color_by is not None and color_by not in umap_df:
        raise ValueError(f"UMAP color column not found: {color_by!r}")
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    if color_by is None:
        ax.scatter(umap_df["UMAP_1"], umap_df["UMAP_2"], s=dot_size, alpha=alpha)
    elif pd.api.types.is_numeric_dtype(umap_df[color_by]):
        points = ax.scatter(
            umap_df["UMAP_1"],
            umap_df["UMAP_2"],
            c=umap_df[color_by],
            cmap=cmap,
            s=dot_size,
            alpha=alpha,
        )
        fig.colorbar(points, ax=ax, label=color_by)
    else:
        groups = list(pd.unique(umap_df[color_by].astype(str)))
        colors = plt.get_cmap(cmap, len(groups))
        for index, group in enumerate(groups):
            mask = umap_df[color_by].astype(str) == group
            ax.scatter(
                umap_df.loc[mask, "UMAP_1"],
                umap_df.loc[mask, "UMAP_2"],
                color=colors(index),
                label=group,
                s=dot_size,
                alpha=alpha,
            )
        if legend_outside:
            ax.legend(
                title=color_by,
                markerscale=2,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0,
            )
            # Reserve figure space so the side legend does not cover the embedding.
            fig.subplots_adjust(right=0.72)
        else:
            ax.legend(title=color_by, markerscale=2)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("MSI UMAP")
    return fig, ax


def plot_cluster_classification(
    cluster_df: pd.DataFrame,
    cmap: str = "tab20",
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
):
    """Plot pixel-wise assignments from either k-means clustering function.
    
    :param cluster_df: Dataframe with x, y, cluster, and optional sample columns
    :param cmap: Matplotlib colormap name used for clusters
    :param ncols: Number of subplot columns when multiple samples are present
    :param figsize: Optional figure size
    :return: Tuple of matplotlib figure and active axes array"""
    required = {"x", "y", "cluster"}
    if not required.issubset(cluster_df.columns):
        missing = required - set(cluster_df.columns)
        raise ValueError(f"cluster_df is missing required columns: {sorted(missing)}")
    if cluster_df.empty:
        raise ValueError("cluster_df is empty.")

    if "sample" not in cluster_df.columns:
        data_groups = [("sample", cluster_df)]
    else:
        data_groups = list(cluster_df.groupby("sample", sort=False))

    n_samples = len(data_groups)
    if n_samples == 0:
        raise ValueError("cluster_df is empty.")

    ncols = max(1, min(ncols, n_samples))
    nrows = int(np.ceil(n_samples / ncols))

    sample_sizes: list[tuple[float, float]] = []
    for _, sample_df in data_groups:
        x = sample_df["x"].to_numpy(dtype=int)
        y = sample_df["y"].to_numpy(dtype=int)
        pixel_size_x = (
            float(sample_df["pixel_size_x"].iloc[0]) if "pixel_size_x" in sample_df.columns else 1.0
        )
        pixel_size_y = (
            float(sample_df["pixel_size_y"].iloc[0]) if "pixel_size_y" in sample_df.columns else 1.0
        )
        phys_width = (x.max() - x.min() + 1) * pixel_size_x
        phys_height = (y.max() - y.min() + 1) * pixel_size_y
        sample_sizes.append((phys_width, phys_height))

    width_ratios = [1.0] * ncols
    height_ratios = [1.0] * nrows
    for idx, (phys_w, phys_h) in enumerate(sample_sizes):
        row = idx // ncols
        col = idx % ncols
        width_ratios[col] = max(width_ratios[col], phys_w)
        height_ratios[row] = max(height_ratios[row], phys_h)

    if figsize is None:
        figsize = (4.5 * ncols + 1.2, 4.5 * nrows)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        squeeze=False,
        gridspec_kw={"width_ratios": width_ratios, "height_ratios": height_ratios},
        constrained_layout=True,
    )
    flat_axes = axes.ravel()
    cluster_labels = np.sort(pd.unique(cluster_df["cluster"]).astype(int))
    n_cluster_labels = len(cluster_labels)
    cluster_to_idx = {label: idx for idx, label in enumerate(cluster_labels)}
    cluster_colors = _cluster_color_mapping(cluster_labels, cmap=cmap)
    discrete_cmap = mpl.colors.ListedColormap([cluster_colors[label] for label in cluster_labels])
    norm = mpl.colors.BoundaryNorm(np.arange(-0.5, n_cluster_labels + 0.5, 1), n_cluster_labels)
    last_im = None

    for idx, (sample_name, sample_df) in enumerate(data_groups):
        x = sample_df["x"].to_numpy(dtype=int)
        y = sample_df["y"].to_numpy(dtype=int)
        labels = sample_df["cluster"].to_numpy(dtype=int)
        label_idx = np.array([cluster_to_idx[val] for val in labels], dtype=float)
        pixel_size_x = (
            float(sample_df["pixel_size_x"].iloc[0]) if "pixel_size_x" in sample_df.columns else 1.0
        )
        pixel_size_y = (
            float(sample_df["pixel_size_y"].iloc[0]) if "pixel_size_y" in sample_df.columns else 1.0
        )

        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        img = np.full((y_max - y_min + 1, x_max - x_min + 1), np.nan, dtype=float)
        img[y - y_min, x - x_min] = label_idx

        ax = flat_axes[idx]
        masked_img = np.ma.masked_invalid(img)
        extent = (
            (x_min - 0.5) * pixel_size_x,
            (x_max + 0.5) * pixel_size_x,
            (y_min - 0.5) * pixel_size_y,
            (y_max + 0.5) * pixel_size_y,
        )
        last_im = ax.imshow(
            masked_img,
            cmap=discrete_cmap,
            norm=norm,
            interpolation="nearest",
            origin="lower",
            extent=extent,
        )
        ax.set_title(str(sample_name))
        ax.set_xlabel("x (physical units)")
        ax.set_ylabel("y (physical units)")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    for ax in flat_axes[n_samples:]:
        ax.axis("off")

    if last_im is not None:
        cbar = fig.colorbar(
            last_im,
            ax=list(flat_axes[:n_samples]),
            fraction=0.03,
            pad=0.03,
            ticks=np.arange(n_cluster_labels),
        )
        cbar.ax.set_yticklabels([str(label) for label in cluster_labels])
        cbar.set_label("Cluster")

    cluster_df.attrs["cluster_cmap"] = cmap
    cluster_df.attrs["cluster_colors"] = cluster_colors
    return fig, flat_axes[:n_samples]


def mean_spectra_by_cluster(
    cluster_df: pd.DataFrame,
    imzml_paths: str | Path | Sequence[str | Path],
    tic_normalize: bool | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Computes mean spectra for each cluster label in a clustering result.
    
    :param cluster_df: Output dataframe from kmeans_cluster_imzml
    :param imzml_paths: One imzML path or sequence of paths used in clustering
    :param tic_normalize: Whether to TIC-normalize spectra before averaging
    :return: Tuple of m/z axis and mean spectra dataframe"""
    required = {"sample", "x", "y", "cluster"}
    if not required.issubset(cluster_df.columns):
        missing = required - set(cluster_df.columns)
        raise ValueError(f"cluster_df is missing required columns: {sorted(missing)}")
    if cluster_df.empty:
        raise ValueError("cluster_df is empty.")
    if tic_normalize is None:
        tic_normalize = bool(cluster_df.attrs.get("tic_normalized", True))

    paths = _normalize_imzml_paths(imzml_paths)
    sample_to_path: dict[str, Path] = {}
    for path in paths:
        sample = path.stem
        if sample in sample_to_path:
            raise ValueError(
                f"Duplicate sample stem detected: '{sample}'. Use unique filenames for imzML files."
            )
        sample_to_path[sample] = path

    df = cluster_df.copy()
    if "z" not in df.columns:
        df["z"] = 1

    cluster_lookup: dict[tuple[str, int, int, int], int] = {
        (str(row.sample), int(row.x), int(row.y), int(row.z)): int(row.cluster)
        for row in df.itertuples(index=False)
    }

    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    reference_mz: np.ndarray | None = None

    for sample_name in df["sample"].unique():
        if sample_name not in sample_to_path:
            raise ValueError(
                f"Sample '{sample_name}' in cluster_df has no matching imzML path."
            )
        path = sample_to_path[sample_name]
        with warnings.catch_warnings(action="ignore"):
            imzml = ImzMLParser.ImzMLParser(path)

        _validate_file_continuous(imzml)
        for idx, coord in enumerate(imzml.coordinates):
            x, y, z = coord
            key = (sample_name, int(x), int(y), int(z))
            cluster_label = cluster_lookup.get(key)
            if cluster_label is None:
                continue

            mz, intensity = imzml.getspectrum(idx)
            mz = np.asarray(mz)
            intensity = np.asarray(intensity, dtype=float)
            if tic_normalize:
                total = intensity.sum()
                if total > 0:
                    intensity = intensity / total

            if reference_mz is None:
                reference_mz = mz
            elif not np.array_equal(mz, reference_mz):
                raise ValueError(
                    "All spectra used for averaging must share the same m/z axis."
                )

            if cluster_label not in sums:
                sums[cluster_label] = np.zeros_like(intensity, dtype=float)
                counts[cluster_label] = 0
            sums[cluster_label] += intensity
            counts[cluster_label] += 1

    if reference_mz is None or len(sums) == 0:
        raise ValueError("No overlapping spectra found between cluster_df and imzML files.")

    cluster_ids = sorted(sums.keys())
    mean_data = np.column_stack([sums[c] / counts[c] for c in cluster_ids])
    mean_df = pd.DataFrame(mean_data, index=reference_mz, columns=cluster_ids)
    mean_df.index.name = "mz"
    mean_df.attrs["tic_normalized"] = bool(tic_normalize)
    mean_df.attrs["cluster_cmap"] = cluster_df.attrs.get("cluster_cmap", "tab20")
    if "cluster_colors" in cluster_df.attrs:
        # Keep only colors of clusters present in mean_df.
        mean_df.attrs["cluster_colors"] = {
            int(k): v for k, v in cluster_df.attrs["cluster_colors"].items() if int(k) in cluster_ids
        }
    return reference_mz, mean_df


def mean_spectra_by_cluster_zarr(
    cluster_df: pd.DataFrame,
    zarr_paths: str | Path | Sequence[str | Path],
    msi_dataset: str,
    tic_normalize: bool | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Compute mean cluster spectra from selected SpatialData Zarr MSI tables.

    :param cluster_df: Output dataframe from :func:`kmeans_cluster_zarr`
    :param zarr_paths: Zarr path or paths used to create ``cluster_df``
    :param msi_dataset: Display name, label, table key, or TIC key of the MSI dataset
    :param tic_normalize: Whether to TIC-normalize spectra before averaging; by
        default, reuse the setting stored in ``cluster_df``
    :return: Tuple of the m/z axis and a dataframe with one mean spectrum per cluster
    """
    required = {"sample", "x", "y", "cluster"}
    if not required.issubset(cluster_df.columns):
        missing = required - set(cluster_df.columns)
        raise ValueError(f"cluster_df is missing required columns: {sorted(missing)}")
    if cluster_df.empty:
        raise ValueError("cluster_df is empty.")
    if tic_normalize is None:
        tic_normalize = bool(cluster_df.attrs.get("tic_normalized", True))

    paths = _normalize_zarr_paths(zarr_paths)
    sample_tables: dict[str, tuple[Path, object]] = {}
    for path in paths:
        table = _load_zarr_msi_table(path, msi_dataset)
        if "x" not in table.obs or "y" not in table.obs:
            raise ValueError(
                f"MSI table {msi_dataset!r} in {path} must contain obs['x'] and obs['y']."
            )
        display_name = str(
            table.uns.get("coregistration_display_name")
            or table.uns.get("coregistration_dataset_label")
            or msi_dataset
        )
        sample_name = display_name if len(paths) == 1 else f"{path.stem}: {display_name}"
        if sample_name in sample_tables:
            raise ValueError(f"Duplicate Zarr sample name detected: {sample_name!r}.")
        sample_tables[sample_name] = (path, table)

    cluster_rows = cluster_df.copy()
    if "z" not in cluster_rows:
        cluster_rows["z"] = 1

    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    reference_mz: np.ndarray | None = None

    for sample_name in cluster_rows["sample"].astype(str).unique():
        if sample_name not in sample_tables:
            available = ", ".join(sample_tables)
            raise ValueError(
                f"Sample {sample_name!r} in cluster_df has no matching Zarr table. "
                f"Available samples: {available}"
            )
        path, table = sample_tables[sample_name]
        local_mz = _table_mz_axis(table)
        if reference_mz is None:
            reference_mz = local_mz
        elif not np.array_equal(local_mz, reference_mz):
            raise ValueError(
                "All selected Zarr MSI tables must share the same m/z axis for averaging."
            )

        sample_rows = cluster_rows[cluster_rows["sample"].astype(str) == sample_name]
        cluster_lookup = {
            (int(row.x), int(row.y), int(row.z)): int(row.cluster)
            for row in sample_rows.itertuples(index=False)
        }
        x_values = table.obs["x"].to_numpy()
        y_values = table.obs["y"].to_numpy()
        z_values = (
            table.obs["z"].to_numpy()
            if "z" in table.obs
            else np.ones(table.n_obs, dtype=int)
        )
        matched_indices: list[int] = []
        matched_clusters: list[int] = []
        for idx, (x, y, z) in enumerate(zip(x_values, y_values, z_values, strict=True)):
            cluster_label = cluster_lookup.get((int(x), int(y), int(z)))
            if cluster_label is not None:
                matched_indices.append(idx)
                matched_clusters.append(cluster_label)

        if not matched_indices:
            continue
        matrix = _table_spectral_matrix(table)[matched_indices]
        values = matrix.data if sparse.issparse(matrix) else matrix
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Spectral data in {path} contains NaN or infinite values.")
        if tic_normalize:
            matrix = _tic_normalize_matrix(matrix)

        matched_clusters_array = np.asarray(matched_clusters, dtype=int)
        for cluster_label in np.unique(matched_clusters_array):
            mask = matched_clusters_array == cluster_label
            cluster_sum = np.asarray(matrix[mask].sum(axis=0)).reshape(-1)
            if cluster_label not in sums:
                sums[int(cluster_label)] = np.zeros(table.n_vars, dtype=float)
                counts[int(cluster_label)] = 0
            sums[int(cluster_label)] += cluster_sum
            counts[int(cluster_label)] += int(np.count_nonzero(mask))

    if reference_mz is None or not sums:
        raise ValueError("No overlapping spectra found between cluster_df and Zarr tables.")

    cluster_ids = sorted(sums)
    mean_data = np.column_stack([sums[label] / counts[label] for label in cluster_ids])
    mean_df = pd.DataFrame(mean_data, index=reference_mz, columns=cluster_ids)
    mean_df.index.name = "mz"
    mean_df.attrs["tic_normalized"] = bool(tic_normalize)
    mean_df.attrs["source_format"] = "zarr"
    mean_df.attrs["msi_dataset"] = str(msi_dataset)
    mean_df.attrs["cluster_cmap"] = cluster_df.attrs.get("cluster_cmap", "tab20")
    if "cluster_colors" in cluster_df.attrs:
        mean_df.attrs["cluster_colors"] = {
            int(key): value
            for key, value in cluster_df.attrs["cluster_colors"].items()
            if int(key) in cluster_ids
        }
    return reference_mz, mean_df


def plot_mean_spectra_by_cluster(
    mz_axis: np.ndarray,
    mean_spectra_df: pd.DataFrame,
    ax=None,
    separate_axes: bool = True,
    ncols: int = 2,
    linewidth: float = 1.2,
    cmap: str = "tab20",
    cluster_colors: dict[int, tuple[float, float, float, float]] | None = None,
    label_peaks: bool = True,
    max_peak_labels: int = 8,
    min_rel_prominence: float = 0.05,
):
    """Plot mean spectra returned by either cluster-averaging function.
    
    :param mz_axis: m/z axis values
    :param mean_spectra_df: Mean spectra dataframe with cluster labels as columns
    :param ax: Optional target axes when drawing all spectra on one axis
    :param separate_axes: Whether to draw each cluster on a separate axis
    :param ncols: Number of subplot columns when separate axes are used
    :param linewidth: Width of spectrum lines
    :param cmap: Matplotlib colormap name for cluster colors
    :param cluster_colors: Optional mapping of cluster labels to RGBA colors
    :param label_peaks: Whether to annotate prominent peaks
    :param max_peak_labels: Maximum number of peaks to label per spectrum
    :param min_rel_prominence: Minimum relative prominence required for labeling
    :return: Tuple of matplotlib figure and active axes array"""
    if mean_spectra_df.empty:
        raise ValueError("mean_spectra_df is empty.")

    labels = [int(c) for c in mean_spectra_df.columns]
    if cluster_colors is None:
        if "cluster_colors" in mean_spectra_df.attrs:
            cluster_colors = {
                int(k): v for k, v in mean_spectra_df.attrs["cluster_colors"].items()
            }
        else:
            cmap_to_use = mean_spectra_df.attrs.get("cluster_cmap", cmap)
            cluster_colors = _cluster_color_mapping(labels, cmap=cmap_to_use)

    if not separate_axes:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4.5))
        else:
            fig = ax.figure
        for cluster_label in labels:
            y_vals = mean_spectra_df[cluster_label].to_numpy()
            ax.vlines(
                mz_axis,
                0,
                y_vals,
                linewidth=linewidth,
                label=f"Cluster {cluster_label}",
                color=cluster_colors[cluster_label],
            )
            if label_peaks:
                _annotate_peaks(
                    ax=ax,
                    mz_axis=mz_axis,
                    intensities=y_vals,
                    color=cluster_colors[cluster_label],
                    max_labels=max_peak_labels,
                    min_rel_prominence=min_rel_prominence,
                )
        ax.set_xlabel("m/z")
        ax.set_ylabel("Mean Intensity")
        ax.legend(ncols=2)
        ax.set_title("Mean Spectra by Cluster")
        return fig, np.array([ax], dtype=object)

    cluster_labels = labels
    n_clusters = len(cluster_labels)
    ncols = max(1, min(ncols, n_clusters))
    nrows = int(np.ceil(n_clusters / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 3 * nrows),
        squeeze=False,
        sharex=True,
        constrained_layout=True,
    )
    flat_axes = axes.ravel()

    for idx, cluster_label in enumerate(cluster_labels):
        axis = flat_axes[idx]
        y_vals = mean_spectra_df[cluster_label].to_numpy()
        axis.vlines(
            mz_axis,
            0,
            y_vals,
            linewidth=linewidth,
            color=cluster_colors[cluster_label],
        )
        if label_peaks:
            _annotate_peaks(
                ax=axis,
                mz_axis=mz_axis,
                intensities=y_vals,
                color=cluster_colors[cluster_label],
                max_labels=max_peak_labels,
                min_rel_prominence=min_rel_prominence,
            )
        axis.set_title(f"Cluster {cluster_label}")
        axis.set_ylabel("Mean Intensity")
        axis.grid(alpha=0.2)

    for axis in flat_axes[n_clusters:]:
        axis.axis("off")

    for axis in flat_axes[:n_clusters]:
        axis.set_xlabel("m/z")

    return fig, flat_axes[:n_clusters]
