import warnings
from collections.abc import Sequence
from pyimzml import ImzMLParser
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
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
    if isinstance(n_clusters, str) and n_clusters != "auto":
        raise ValueError("n_clusters must be an integer >= 1 or 'auto'.")
    if isinstance(n_clusters, int) and n_clusters < 1:
        raise ValueError("n_clusters must be at least 1.")
    if min_cluster_fraction < 0:
        raise ValueError("min_cluster_fraction must be >= 0.")
    if min_cluster_size < 1:
        raise ValueError("min_cluster_size must be >= 1.")

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
        if tic_normalize:
            total = local_intensity.sum()
            if total > 0:
                local_intensity = local_intensity / total
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
            if tic_normalize:
                total = intensity.sum()
                if total > 0:
                    intensity = intensity / total
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

    data = np.vstack(spectra_blocks)
    n_pixels = data.shape[0]
    if n_pixels < 1:
        raise ValueError("No spectra found to cluster.")

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

    # In auto mode, drop tiny clusters by reassigning their pixels to the nearest
    # remaining centroid, then relabel to contiguous 1..k.
    if auto_mode:
        counts = np.bincount(labels0, minlength=initial_k)
        min_size_threshold = max(min_cluster_size, int(np.ceil(min_cluster_fraction * n_pixels)))
        keep = np.where(counts >= min_size_threshold)[0]
        if keep.size == 0:
            keep = np.array([int(np.argmax(counts))], dtype=int)

        drop = np.setdiff1d(np.arange(initial_k), keep, assume_unique=True)
        labels_adj = labels0.copy()
        if drop.size > 0:
            dropped_mask = np.isin(labels_adj, drop)
            if np.any(dropped_mask):
                kept_centers = model.cluster_centers_[keep]
                distances = np.sum(
                    (data[dropped_mask, None, :] - kept_centers[None, :, :]) ** 2,
                    axis=2,
                )
                nearest_keep_idx = np.argmin(distances, axis=1)
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
    if auto_mode:
        df.attrs["min_cluster_fraction"] = float(min_cluster_fraction)
        df.attrs["min_cluster_size"] = int(min_cluster_size)
        df.attrs["min_size_threshold_used"] = int(
            max(min_cluster_size, int(np.ceil(min_cluster_fraction * n_pixels)))
        )
    return df


def plot_cluster_classification(
    cluster_df: pd.DataFrame,
    cmap: str = "tab20",
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
):
    """Plots pixel-wise cluster assignments from kmeans_cluster_imzml using imshow.
    
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
    """Plots mean cluster spectra returned by mean_spectra_by_cluster.
    
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
