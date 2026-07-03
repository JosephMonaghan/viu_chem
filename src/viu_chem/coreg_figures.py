from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
from scipy import ndimage
from spatialdata.transformations import get_transformation
import zarr

from .msi_coregistration import (
    CoregistrationDataset,
    REFERENCE_CHANNEL_COLOR_PRESETS,
    _annotation_mask_from_transformed_geometries,
    _fallback_reference_channel_color,
    _reference_channel_image,
    _resolve_msi_dataset_keys,
    _sample_reference_values_at_msi_pixels,
    _xy_matrix_from_transform,
)


@dataclass
class CoregisteredImage:
    data: np.ndarray
    label: str
    image_key: str | None = None
    channel_index: int | None = None
    mz: float | None = None
    actual_mz: float | None = None
    feature_indices: np.ndarray | None = None
    contrast_limits: tuple[float, float] | None = None


def _display_limits_for_coregistered_image(
    img: np.ndarray,
    low_pct: float = 1.0,
    high_pct: float = 99.8,
    *,
    positive_only: bool = True,
) -> tuple[float, float]:
    if np.ma.isMaskedArray(img):
        finite = np.asarray(img.compressed(), dtype=float)
    else:
        vals = np.asarray(img, dtype=float)
        finite = vals[np.isfinite(vals)]
    if positive_only:
        finite = finite[finite > 0]
    if finite.size == 0:
        return (0.0, 1.0)
    lo = float(np.percentile(finite, low_pct))
    hi = float(np.percentile(finite, high_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
    if hi <= lo:
        hi = lo + 1e-9
    return lo, hi


def mask_low_intensity_pixels(
    img: np.ndarray,
    *,
    low: float | None = None,
    low_pct: float = 1.0,
    high_pct: float = 99.8,
    positive_only: bool = True,
) -> tuple[np.ma.MaskedArray, tuple[float, float]]:
    limits = _display_limits_for_coregistered_image(
        img,
        low_pct=low_pct,
        high_pct=high_pct,
        positive_only=positive_only,
    )
    low_value = limits[0] if low is None else float(low)
    arr = np.asarray(img, dtype=float)
    return np.ma.masked_where(~np.isfinite(arr) | (arr <= low_value), arr), limits


def _image_transform_xy_or_identity(sdata, image_key: str, registered_cs: str) -> np.ndarray:
    try:
        transform = get_transformation(sdata.images[image_key], to_coordinate_system=registered_cs)
        return _xy_matrix_from_transform(transform)
    except Exception:
        return np.eye(3, dtype=float)


def _reference_grid_for_coregistered_arrays(
    dataset: CoregistrationDataset,
    reference_key: str | None,
) -> tuple[str | None, tuple[int, int], np.ndarray]:
    if reference_key is None:
        if dataset.reference_image_keys:
            reference_key = dataset.reference_image_keys[0]
    elif reference_key not in dataset.sdata.images:
        raise KeyError(f"Reference image {reference_key!r} was not found.")

    if reference_key is None:
        return None, (dataset.ny, dataset.nx), np.eye(3, dtype=float)

    reference_img = np.asarray(dataset.sdata.images[reference_key])
    reference_attrs = getattr(dataset.sdata.images[reference_key], "attrs", {})
    reference_dims = tuple(getattr(dataset.sdata.images[reference_key], "dims", ()))
    source_channels = int(reference_attrs.get("source_channels", 0)) if isinstance(reference_attrs, Mapping) else 0
    if reference_img.ndim == 2:
        output_shape = reference_img.shape
    elif reference_img.ndim == 3:
        if reference_dims == ("c", "y", "x"):
            output_shape = tuple(reference_img.shape[-2:])
        elif reference_dims == ("y", "x", "c"):
            output_shape = tuple(reference_img.shape[:2])
        elif source_channels > 4 and reference_img.shape[0] == source_channels:
            output_shape = tuple(reference_img.shape[-2:])
        elif source_channels > 4 and reference_img.shape[-1] == source_channels:
            output_shape = tuple(reference_img.shape[:2])
        elif reference_img.shape[0] <= 4 and reference_img.shape[-1] > 4:
            output_shape = tuple(reference_img.shape[-2:])
        elif reference_img.shape[-1] <= 4:
            output_shape = tuple(reference_img.shape[:2])
        else:
            output_shape = tuple(reference_img.shape[-2:])
    else:
        raise ValueError(f"Reference image {reference_key!r} has unsupported shape {reference_img.shape}.")

    output_transform_xy = _image_transform_xy_or_identity(dataset.sdata, reference_key, dataset.registered_cs)
    return reference_key, (int(output_shape[0]), int(output_shape[1])), output_transform_xy


def _resample_to_output_grid(
    img: np.ndarray,
    *,
    source_transform_xy: np.ndarray,
    output_transform_xy: np.ndarray,
    output_shape: tuple[int, int],
    order: int = 1,
    cval: float = np.nan,
) -> np.ndarray:
    source_to_output_xy = np.linalg.inv(np.asarray(source_transform_xy, dtype=float)) @ np.asarray(output_transform_xy, dtype=float)
    matrix_yx = np.array(
        [
            [source_to_output_xy[1, 1], source_to_output_xy[1, 0]],
            [source_to_output_xy[0, 1], source_to_output_xy[0, 0]],
        ],
        dtype=float,
    )
    offset_yx = np.array([source_to_output_xy[1, 2], source_to_output_xy[0, 2]], dtype=float)
    return ndimage.affine_transform(
        np.asarray(img, dtype=float),
        matrix=matrix_yx,
        offset=offset_yx,
        output_shape=tuple(int(v) for v in output_shape),
        order=int(order),
        mode="constant",
        cval=float(cval),
    )


def get_coregistered_msi_mask_image(
    zarr_path: str | Path | CoregistrationDataset,
    pixel_mask: np.ndarray,
    *,
    msi_dataset: str | None = None,
    reference_key: str | None = "hne",
    registered_cs: str = "registered",
    table_key: str | None = None,
    tic_key: str | None = None,
) -> CoregisteredImage:
    if not isinstance(zarr_path, CoregistrationDataset):
        table_key, tic_key = _resolve_msi_dataset_keys(
            zarr_path,
            msi_dataset=msi_dataset,
            table_key=table_key,
            tic_key=tic_key,
        )
    dataset = (
        zarr_path
        if isinstance(zarr_path, CoregistrationDataset)
        else CoregistrationDataset(zarr_path, registered_cs=registered_cs, table_key=table_key, tic_key=tic_key)
    )
    mask = np.asarray(pixel_mask, dtype=bool)
    if mask.shape == (dataset.ny, dataset.nx):
        raw_img = mask.astype(float)
    else:
        mask = mask.ravel()
        if mask.shape[0] != dataset.x_coords.shape[0]:
            raise ValueError("pixel_mask must match either the MSI raster shape or the number of MSI spectra.")
        raw_img = np.zeros((dataset.ny, dataset.nx), dtype=float)
        raw_img[dataset.y_coords[mask], dataset.x_coords[mask]] = 1.0

    _, output_shape, output_transform_xy = _reference_grid_for_coregistered_arrays(dataset, reference_key)
    source_transform_xy, _found = dataset.load_saved_registration_if_available()
    resampled = _resample_to_output_grid(
        raw_img,
        source_transform_xy=source_transform_xy,
        output_transform_xy=output_transform_xy,
        output_shape=output_shape,
        order=0,
        cval=0.0,
    )
    data = np.ma.masked_where(~np.isfinite(resampled) | (resampled < 0.5), resampled >= 0.5)
    return CoregisteredImage(
        data=data,
        label="MSI mask",
        contrast_limits=(0.0, 1.0),
    )


def get_coregistered_reference_image(
    zarr_path: str | Path | CoregistrationDataset,
    *,
    reference_key: str | None = "hne",
    channel_index: int = 0,
    output_reference_key: str | None = None,
    mask_low: bool = True,
    low_pct: float = 1.0,
    high_pct: float = 99.8,
    registered_cs: str = "registered",
) -> CoregisteredImage:
    dataset = zarr_path if isinstance(zarr_path, CoregistrationDataset) else CoregistrationDataset(zarr_path, registered_cs=registered_cs)
    if reference_key is None:
        if not dataset.reference_image_keys:
            raise ValueError("No reference images are available.")
        reference_key = dataset.reference_image_keys[0]
    if reference_key not in dataset.sdata.images:
        raise KeyError(f"Reference image {reference_key!r} was not found.")

    _, output_shape, output_transform_xy = _reference_grid_for_coregistered_arrays(dataset, output_reference_key or reference_key)
    img = _reference_channel_image(dataset.sdata, reference_key, int(channel_index))
    source_transform_xy = _image_transform_xy_or_identity(dataset.sdata, reference_key, dataset.registered_cs)
    resampled = _resample_to_output_grid(
        img,
        source_transform_xy=source_transform_xy,
        output_transform_xy=output_transform_xy,
        output_shape=output_shape,
        order=1,
    )
    if mask_low:
        data, limits = mask_low_intensity_pixels(resampled, low_pct=low_pct, high_pct=high_pct)
    else:
        data = resampled
        limits = _display_limits_for_coregistered_image(resampled, low_pct=low_pct, high_pct=high_pct)
    return CoregisteredImage(
        data=data,
        label=f"{reference_key} ch {int(channel_index) + 1}",
        image_key=reference_key,
        channel_index=int(channel_index),
        contrast_limits=limits,
    )


def get_coregistered_ion_image(
    zarr_path: str | Path | CoregistrationDataset,
    target_mz: float,
    *,
    msi_dataset: str | None = None,
    ppm_tolerance: float = 5.0,
    reference_key: str | None = "hne",
    normalize_to_tic: bool = True,
    mask_low: bool = True,
    low_pct: float = 1.0,
    high_pct: float = 99.8,
    registered_cs: str = "registered",
    table_key: str | None = None,
    tic_key: str | None = None,
    resample_order: int = 0,
) -> CoregisteredImage:
    if not isinstance(zarr_path, CoregistrationDataset):
        table_key, tic_key = _resolve_msi_dataset_keys(
            zarr_path,
            msi_dataset=msi_dataset,
            table_key=table_key,
            tic_key=tic_key,
        )
    dataset = (
        zarr_path
        if isinstance(zarr_path, CoregistrationDataset)
        else CoregistrationDataset(zarr_path, registered_cs=registered_cs, table_key=table_key, tic_key=tic_key)
    )
    _, output_shape, output_transform_xy = _reference_grid_for_coregistered_arrays(dataset, reference_key)
    indices = dataset.find_feature_indices_from_mz(float(target_mz), float(ppm_tolerance))
    if indices.size == 0:
        idx, _ppm_error = dataset.find_feature_idx_from_mz(float(target_mz), float("inf"))
        if idx is None:
            raise ValueError(f"No m/z feature found near {target_mz:g}.")
        indices = np.array([idx], dtype=int)
    raw_img = dataset.reconstruct_ion_image(indices, normalize_to_tic=bool(normalize_to_tic))
    source_transform_xy, _found = dataset.load_saved_registration_if_available()
    resampled = _resample_to_output_grid(
        raw_img,
        source_transform_xy=source_transform_xy,
        output_transform_xy=output_transform_xy,
        output_shape=output_shape,
        order=int(resample_order),
    )
    if mask_low:
        data, limits = mask_low_intensity_pixels(resampled, low_pct=low_pct, high_pct=high_pct)
    else:
        data = resampled
        limits = _display_limits_for_coregistered_image(resampled, low_pct=low_pct, high_pct=high_pct)
    nearest_idx = int(indices[np.argmin(np.abs(dataset.mz_values[indices] - float(target_mz)))])
    return CoregisteredImage(
        data=data,
        label=f"m/z {float(target_mz):.4f}",
        mz=float(target_mz),
        actual_mz=float(dataset.mz_values[nearest_idx]),
        feature_indices=indices.astype(int),
        contrast_limits=limits,
    )


def get_coregistered_image_layers(
    zarr_path: str | Path,
    *,
    mzs: Iterable[float],
    msi_dataset: str | None = None,
    reference_key: str | None = "hne",
    reference_channel_index: int = 0,
    ppm_tolerance: float = 5.0,
    normalize_to_tic: bool = True,
    mask_low: bool = True,
    low_pct: float = 1.0,
    high_pct: float = 99.8,
    registered_cs: str = "registered",
    table_key: str | None = None,
    tic_key: str | None = None,
    ion_resample_order: int = 0,
) -> dict[str, Any]:
    table_key, tic_key = _resolve_msi_dataset_keys(
        zarr_path,
        msi_dataset=msi_dataset,
        table_key=table_key,
        tic_key=tic_key,
    )
    dataset = CoregistrationDataset(zarr_path, registered_cs=registered_cs, table_key=table_key, tic_key=tic_key)
    chosen_reference_key = reference_key
    if chosen_reference_key is not None and chosen_reference_key not in dataset.sdata.images:
        if dataset.reference_image_keys:
            chosen_reference_key = dataset.reference_image_keys[0]
        else:
            chosen_reference_key = None
    reference = (
        get_coregistered_reference_image(
            dataset,
            reference_key=chosen_reference_key,
            channel_index=reference_channel_index,
            output_reference_key=chosen_reference_key,
            mask_low=mask_low,
            low_pct=low_pct,
            high_pct=high_pct,
        )
        if chosen_reference_key is not None
        else None
    )
    ions = [
        get_coregistered_ion_image(
            dataset,
            float(mz),
            ppm_tolerance=ppm_tolerance,
            reference_key=chosen_reference_key,
            normalize_to_tic=normalize_to_tic,
            mask_low=mask_low,
            low_pct=low_pct,
            high_pct=high_pct,
            resample_order=ion_resample_order,
        )
        for mz in mzs
    ]
    return {
        "dataset": dataset,
        "reference": reference,
        "ions": ions,
        "reference_key": chosen_reference_key,
    }


def _read_saved_if_display_settings(dataset: CoregistrationDataset, reference_key: str) -> dict[str, Any]:
    try:
        root = zarr.open_group(dataset.zarr_path, mode="r", use_consolidated=False)
        raw = root["images"][reference_key].attrs.get("if_display_settings", {})
    except Exception:
        raw = {}
    if not isinstance(raw, dict):
        image = dataset.sdata.images[reference_key]
        attrs = getattr(image, "attrs", {})
        raw = attrs.get("if_display_settings", {}) if hasattr(attrs, "get") else {}
    return dict(raw) if isinstance(raw, dict) else {}


def _channel_saved_settings(saved_settings: Mapping[Any, Any], index: int) -> dict[str, Any]:
    raw = saved_settings.get(str(index), saved_settings.get(index, {}))
    return dict(raw) if isinstance(raw, dict) else {}


def _apply_reference_overrides(
    saved_settings: dict[str, Any],
    overrides: Mapping[Any, Mapping[str, Any]] | None,
    channel_names: list[str],
) -> dict[str, Any]:
    if not overrides:
        return saved_settings
    merged = {str(key): dict(value) if isinstance(value, Mapping) else value for key, value in saved_settings.items()}
    name_to_index = {str(name).strip().lower(): idx for idx, name in enumerate(channel_names) if str(name).strip()}
    for key, value in overrides.items():
        if not isinstance(value, Mapping):
            continue
        if isinstance(key, int) or str(key).isdigit():
            idx = int(key)
        else:
            idx = name_to_index.get(str(key).strip().lower())
            if idx is None:
                continue
        current = dict(merged.get(str(idx), {}))
        current.update(dict(value))
        merged[str(idx)] = current
    return merged


def _metadata_rgb(raw_colors: list[Any], index: int, saved: Mapping[str, Any] | None = None) -> tuple[float, float, float]:
    color_choice = str((saved or {}).get("color_choice", "metadata")).strip().lower()
    if color_choice and color_choice != "metadata":
        return REFERENCE_CHANNEL_COLOR_PRESETS.get(color_choice, REFERENCE_CHANNEL_COLOR_PRESETS["white"])
    if index < len(raw_colors) and isinstance(raw_colors[index], (list, tuple)) and len(raw_colors[index]) == 3:
        rgb = np.asarray(raw_colors[index], dtype=float)
        if np.nanmax(rgb) > 1.5:
            rgb = rgb / 255.0
        if np.any(rgb > 0.05):
            return tuple(np.clip(rgb, 0.0, 1.0))
    return _fallback_reference_channel_color(index)


def _saved_contrast_limits(saved: Mapping[str, Any]) -> tuple[float, float] | None:
    mode = str(saved.get("contrast_mode", "percentile")).strip().lower()
    if mode not in {"absolute", "intensity", "limits"}:
        return None
    limits = saved.get("contrast_limits", ())
    if isinstance(limits, (list, tuple)) and len(limits) >= 2:
        low, high = float(limits[0]), float(limits[1])
        if np.isfinite(low) and np.isfinite(high) and high > low:
            return low, high
    return None


def _saved_contrast_percentiles(saved: Mapping[str, Any]) -> tuple[float, float]:
    percentiles = saved.get("contrast_percentiles", ())
    if isinstance(percentiles, (list, tuple)) and len(percentiles) >= 2:
        low, high = float(percentiles[0]), float(percentiles[1])
        if np.isfinite(low) and np.isfinite(high) and high > low:
            return low, high
    return 1.0, 99.8


def _normalize_for_rgb(
    arr: np.ndarray,
    *,
    limits: tuple[float, float] | None = None,
    percentiles: tuple[float, float] = (1.0, 99.8),
    gamma: float = 1.0,
) -> np.ndarray:
    data = np.asarray(arr, dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros(data.shape, dtype=float)
    if limits is None:
        positive = finite[finite > 0]
        if positive.size:
            finite = positive
        lo, hi = np.percentile(finite, percentiles)
    else:
        lo, hi = limits
    if hi <= lo:
        hi = lo + 1e-9
    normalized = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    if np.isfinite(gamma) and gamma > 0 and not np.isclose(gamma, 1.0):
        normalized = normalized ** (1.0 / gamma)
    return normalized


def _reference_channel_rgba(normalized: np.ndarray, color: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    t = np.clip(np.asarray(normalized, dtype=float), 0.0, 1.0)
    rgb_factor = np.where(t <= 0.5, 0.70 * t, 0.35 + 1.30 * (t - 0.5))
    alpha = np.where(t <= 0.5, 1.20 * t, 0.60 + 0.80 * (t - 0.5))
    rgb = rgb_factor[..., np.newaxis] * np.asarray(color, dtype=float)
    return np.clip(rgb, 0.0, 1.0), np.clip(alpha, 0.0, 1.0)


def _fluorescence_rgb_image(
    arr: np.ndarray,
    raw_colors: list[Any],
    saved_settings: Mapping[Any, Any],
    *,
    respect_saved_visibility: bool,
    blend_mode: str,
) -> np.ndarray:
    rgb = np.zeros(arr.shape[1:] + (3,), dtype=float)
    for idx, channel in enumerate(arr):
        saved = _channel_saved_settings(saved_settings, idx)
        if respect_saved_visibility and saved and not bool(saved.get("visible", True)):
            continue
        color = np.asarray(_metadata_rgb(raw_colors, idx, saved=saved), dtype=float)
        normalized = _normalize_for_rgb(
            channel,
            limits=_saved_contrast_limits(saved),
            percentiles=_saved_contrast_percentiles(saved),
            gamma=float(saved.get("gamma", 1.0) or 1.0),
        )
        if str(blend_mode).lower() == "additive":
            rgb += normalized[..., np.newaxis] * color
            continue
        src_rgb, src_alpha = _reference_channel_rgba(normalized, tuple(color))
        rgb = src_rgb * src_alpha[..., np.newaxis] + rgb * (1.0 - src_alpha[..., np.newaxis])
    return np.clip(rgb, 0.0, 1.0)


def reference_rgb_composite(
    zarr_path: str | Path | CoregistrationDataset,
    *,
    reference_key: str = "hne",
    channel_overrides: Mapping[Any, Mapping[str, Any]] | None = None,
    respect_saved_visibility: bool = True,
    blend_mode: str = "translucent",
    registered_cs: str = "registered",
) -> np.ndarray:
    dataset = zarr_path if isinstance(zarr_path, CoregistrationDataset) else CoregistrationDataset(zarr_path, registered_cs=registered_cs)
    image = dataset.sdata.images[reference_key]
    arr = np.asarray(image)
    dims = tuple(getattr(image, "dims", ()))
    attrs = getattr(image, "attrs", {})
    raw_colors = attrs.get("channel_colors", []) if hasattr(attrs, "get") else []
    raw_names = attrs.get("channel_names", []) if hasattr(attrs, "get") else []
    source_channels = int(attrs.get("source_channels", 0)) if hasattr(attrs, "get") else 0
    saved_settings = _apply_reference_overrides(
        _read_saved_if_display_settings(dataset, reference_key),
        channel_overrides,
        [str(name) for name in raw_names],
    )

    if arr.ndim == 2:
        saved = _channel_saved_settings(saved_settings, 0)
        gray = _normalize_for_rgb(
            arr,
            limits=_saved_contrast_limits(saved),
            percentiles=_saved_contrast_percentiles(saved),
            gamma=float(saved.get("gamma", 1.0) or 1.0),
        )
        return np.dstack([gray, gray, gray])

    if arr.ndim != 3:
        raise ValueError(f"Unsupported reference image shape: {arr.shape}")

    inferred_channels = 0
    if source_channels > 4:
        inferred_channels = source_channels
    elif dims == ("c", "y", "x") and arr.shape[0] > 4:
        inferred_channels = int(arr.shape[0])
    elif dims == ("y", "x", "c") and arr.shape[-1] > 4:
        inferred_channels = int(arr.shape[-1])
    elif raw_colors and dims == ("c", "y", "x") and arr.shape[0] == len(raw_colors):
        inferred_channels = int(arr.shape[0])
    elif raw_colors and dims == ("y", "x", "c") and arr.shape[-1] == len(raw_colors):
        inferred_channels = int(arr.shape[-1])
    elif len(raw_colors) > 4 and arr.shape[0] == len(raw_colors):
        inferred_channels = int(arr.shape[0])
    elif len(raw_colors) > 4 and arr.shape[-1] == len(raw_colors):
        inferred_channels = int(arr.shape[-1])

    if inferred_channels > 4 or (raw_colors and inferred_channels > 0):
        channel_first = np.moveaxis(arr, -1, 0) if dims == ("y", "x", "c") or arr.shape[-1] == inferred_channels else arr
        return _fluorescence_rgb_image(
            channel_first[:inferred_channels],
            raw_colors,
            saved_settings,
            respect_saved_visibility=respect_saved_visibility,
            blend_mode=blend_mode,
        )

    if dims == ("c", "y", "x") or (arr.shape[0] <= 4 and arr.shape[-1] > 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] == 1:
        saved = _channel_saved_settings(saved_settings, 0)
        gray = _normalize_for_rgb(
            arr[..., 0],
            limits=_saved_contrast_limits(saved),
            percentiles=_saved_contrast_percentiles(saved),
            gamma=float(saved.get("gamma", 1.0) or 1.0),
        )
        return np.dstack([gray, gray, gray])

    rgb = np.asarray(arr[..., :3], dtype=float)
    if np.nanmax(rgb) > 1.5:
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0)


def _transparent_cmap(name: str):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad((0, 0, 0, 0))
    cmap.set_under((0, 0, 0, 0))
    return cmap


def _format_cbar_value(value: float) -> str:
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) >= 1e4 or abs(value) < 1e-3:
        return f"{value:.1e}"
    return f"{value:.3g}"


def _global_positive_limits(images: Iterable[np.ndarray], low_pct: float = 1.0, high_pct: float = 99.8) -> tuple[float, float]:
    values = []
    for image in images:
        if np.ma.isMaskedArray(image):
            arr = np.asarray(image.compressed(), dtype=float)
        else:
            arr = np.asarray(image, dtype=float).ravel()
            arr = arr[np.isfinite(arr)]
        arr = arr[arr > 0]
        if arr.size:
            values.append(arr)
    if not values:
        return 0.0, 1.0
    pooled = np.concatenate(values)
    low, high = np.percentile(pooled, [low_pct, high_pct])
    if high <= low:
        high = low + 1e-9
    return float(low), float(high)


def _mask_at_or_below(image: np.ndarray, threshold: float) -> np.ma.MaskedArray:
    arr = np.ma.asarray(image, dtype=float)
    return np.ma.masked_where(np.ma.getmaskarray(arr) | ~np.isfinite(arr) | (arr <= threshold), arr)


def _add_ion_colorbar(
    cbar_ax,
    image_artist,
    limits: tuple[float, float],
    label: str,
    *,
    background_color: str,
    foreground_color: str,
    label_size: float,
    tick_size: float,
    extend: bool,
):
    cbar_ax.set_facecolor(background_color)
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    triangle_top = 0.97
    bar_top = 0.88 if extend else 1.0
    cbar_ax.imshow(gradient, aspect="auto", origin="lower", extent=(0, 1, 0, bar_top), cmap=image_artist.cmap)
    if extend:
        cbar_ax.add_patch(
            MplPolygon(
                [(0, bar_top), (0.5, triangle_top), (1, bar_top)],
                facecolor=image_artist.cmap(1.0),
                edgecolor="none",
            )
        )
    for spine in cbar_ax.spines.values():
        spine.set_visible(False)
    cbar_ax.set_xlim(0, 1)
    cbar_ax.set_ylim(0, 1)
    cbar_ax.set_xticks([])
    cbar_ax.yaxis.set_ticks_position("right")
    cbar_ax.yaxis.set_label_position("right")
    tick_labels = [
        _format_cbar_value(limits[0]),
        _format_cbar_value((limits[0] + limits[1]) / 2),
        f"{_format_cbar_value(limits[1])}+" if extend else _format_cbar_value(limits[1]),
    ]
    cbar_ax.set_yticks([0, bar_top / 2, bar_top], labels=tick_labels, color=foreground_color)
    cbar_ax.tick_params(axis="y", color=foreground_color, labelcolor=foreground_color, labelsize=tick_size, length=3, pad=3)
    cbar_ax.set_ylabel(label, color=foreground_color, rotation=270, va="center", labelpad=15, fontsize=label_size)


def _positive_float(value: Any) -> float:
    try:
        result = float(value)
    except Exception:
        return np.nan
    return result if np.isfinite(result) and result > 0 else np.nan


def _reference_pixel_size_um(dataset: CoregistrationDataset, reference_key: str) -> tuple[float, float]:
    image = dataset.sdata.images[reference_key]
    attrs = getattr(image, "attrs", {})
    x_um = _positive_float(attrs.get("pixel_size_x_um")) if hasattr(attrs, "get") else np.nan
    y_um = _positive_float(attrs.get("pixel_size_y_um")) if hasattr(attrs, "get") else np.nan
    if not np.isfinite(x_um) or x_um <= 0:
        x_um = 1.0
    if not np.isfinite(y_um) or y_um <= 0:
        y_um = x_um
    return x_um, y_um


def _msi_pixel_size_um(dataset: CoregistrationDataset) -> tuple[float, float] | None:
    raw = dataset.msi_table.uns.get("raw_metadata", {})
    attrs = getattr(dataset.sdata, "attrs", {})
    detection = attrs.get("pixel_size_detection_info", {}) if hasattr(attrs, "get") else {}
    tic_attrs = getattr(dataset.sdata.images[dataset.tic_key], "attrs", {})
    candidates = []
    if isinstance(raw, Mapping):
        candidates.append((raw.get("pixel size x"), raw.get("pixel size y")))
    if isinstance(detection, Mapping):
        candidates.append((detection.get("detected_x_um"), detection.get("detected_y_um")))
    if hasattr(tic_attrs, "get"):
        candidates.append((tic_attrs.get("pixel_size_x_um"), tic_attrs.get("pixel_size_y_um")))
    if hasattr(attrs, "get"):
        candidates.append((attrs.get("pixel_size_x_um"), attrs.get("pixel_size_y_um")))

    for x_value, y_value in candidates:
        x_um = _positive_float(x_value)
        y_um = _positive_float(y_value)
        if np.isfinite(x_um) and np.isfinite(y_um):
            return x_um, y_um
        if np.isfinite(x_um):
            return x_um, x_um
    return None


def _reference_display_pixel_size_from_msi(
    dataset: CoregistrationDataset,
    reference_key: str,
) -> tuple[float, float] | None:
    msi_pixel_size = _msi_pixel_size_um(dataset)
    if msi_pixel_size is None:
        return None

    source_transform_xy, _found = dataset.load_saved_registration_if_available()
    _, _output_shape, output_transform_xy = _reference_grid_for_coregistered_arrays(dataset, reference_key)
    try:
        source_to_output_xy = np.linalg.inv(np.asarray(output_transform_xy, dtype=float)) @ np.asarray(source_transform_xy, dtype=float)
    except Exception:
        return None
    if source_to_output_xy.shape != (3, 3) or not np.all(np.isfinite(source_to_output_xy)):
        return None

    origin = source_to_output_xy @ np.array([0.0, 0.0, 1.0])
    x_step = source_to_output_xy @ np.array([1.0, 0.0, 1.0])
    y_step = source_to_output_xy @ np.array([0.0, 1.0, 1.0])
    x_pixels = float(np.linalg.norm(x_step[:2] - origin[:2]))
    y_pixels = float(np.linalg.norm(y_step[:2] - origin[:2]))
    if not np.isfinite(x_pixels) or x_pixels <= 0 or not np.isfinite(y_pixels) or y_pixels <= 0:
        return None
    return float(msi_pixel_size[0]) / x_pixels, float(msi_pixel_size[1]) / y_pixels


def _scale_bar_length_to_um(length: float, units: str) -> float:
    unit = str(units).strip().lower().replace("µ", "u").replace("μ", "u")
    if unit in {"um", "micron", "microns", "micrometer", "micrometers"}:
        return float(length)
    if unit in {"mm", "millimeter", "millimeters"}:
        return float(length) * 1000.0
    if unit in {"nm", "nanometer", "nanometers"}:
        return float(length) / 1000.0
    raise ValueError("scale_bar_units must be one of 'um', 'mm', or 'nm'.")


def _add_scale_bar(
    ax,
    image_shape: tuple[int, int],
    *,
    length: float,
    units: str,
    pixel_size_um: tuple[float, float],
    location: str,
    color: str,
    linewidth: float,
    fontsize: float,
    pad_fraction: float,
    label_pad_fraction: float,
    box_alpha: float,
):
    height, width = int(image_shape[0]), int(image_shape[1])
    length_um = _scale_bar_length_to_um(float(length), units)
    length_px = length_um / float(pixel_size_um[0])
    if not np.isfinite(length_px) or length_px <= 0:
        raise ValueError("Scale bar length must be positive and finite.")
    max_length_px = max(1.0, width * 0.75)
    if length_px > max_length_px:
        max_length_um = max_length_px * float(pixel_size_um[0])
        raise ValueError(
            f"Scale bar is too wide for the image. Requested {length:g} {units}, "
            f"but the maximum unclipped length is about {max_length_um:g} um."
        )

    location = str(location).strip().lower().replace("_", " ")
    x_pad = max(min(float(pad_fraction), 0.45), 0.0)
    y_pad = max(min(float(pad_fraction), 0.45), 0.0)
    label_pad = max(min(float(label_pad_fraction), 0.25), 0.0)
    length_fraction = length_px / float(width)

    if "left" in location:
        x0 = x_pad
        x1 = x0 + length_fraction
    else:
        x1 = 1.0 - x_pad
        x0 = x1 - length_fraction
    text_x = (x0 + x1) / 2.0

    if "upper" in location or "top" in location:
        y = 1.0 - y_pad
        text_y = y - label_pad
        va = "top"
    else:
        y = y_pad
        text_y = y + label_pad
        va = "bottom"

    label = f"{length:g} {units}"
    transform = ax.transAxes
    ax.plot(
        [x0, x1],
        [y, y],
        color="black",
        linewidth=float(linewidth) + 1.8,
        alpha=max(min(float(box_alpha), 1.0), 0.0),
        solid_capstyle="butt",
        clip_on=False,
        transform=transform,
        zorder=19,
    )
    ax.plot(
        [x0, x1],
        [y, y],
        color=color,
        linewidth=float(linewidth),
        solid_capstyle="butt",
        clip_on=False,
        transform=transform,
        zorder=20,
    )
    ax.text(
        text_x,
        text_y,
        label,
        color=color,
        ha="center",
        va=va,
        fontsize=float(fontsize),
        clip_on=False,
        transform=transform,
        zorder=21,
        bbox={
            "boxstyle": "square,pad=0.18",
            "facecolor": "black",
            "edgecolor": "none",
            "alpha": float(box_alpha),
        },
    )


def _save_figure_for_illustrator(fig, output_path: str | Path, *, dpi: int, background_color: str, save_pad_inches: float):
    """Save a vector-container figure with editable text/vector annotations."""
    vector_rc = {
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(vector_rc):
        fig.savefig(
            Path(output_path).expanduser(),
            dpi=int(dpi),
            facecolor=background_color,
            bbox_inches="tight",
            pad_inches=float(save_pad_inches),
        )


def _iter_output_paths(paths: str | Path | Iterable[str | Path] | None) -> list[Path]:
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        return [Path(paths).expanduser()]
    return [Path(path).expanduser() for path in paths]


def _ion_layer_config(
    layer: Mapping[str, Any],
    *,
    default_msi_dataset: str | None,
    default_cmap: str,
    default_alpha: float,
    default_low_pct: float,
    default_high_pct: float,
    default_normalize_to_tic: bool,
    default_resample_order: int,
) -> dict[str, Any]:
    config = dict(layer)
    if "mz" not in config:
        raise ValueError("Each ion layer must include `mz`.")
    config.setdefault("msi_dataset", default_msi_dataset)
    config.setdefault("label", f"m/z {float(config['mz']):.4f}")
    config.setdefault("cmap", default_cmap)
    config.setdefault("alpha", default_alpha)
    config.setdefault("low_pct", default_low_pct)
    config.setdefault("high_pct", default_high_pct)
    config.setdefault("normalize_to_tic", default_normalize_to_tic)
    config.setdefault("resample_order", default_resample_order)
    return config


def _ion_overlay_from_config(
    zarr_path: str | Path,
    config: Mapping[str, Any],
    *,
    reference_key: str,
    ppm_tolerance: float,
    registered_cs: str,
) -> tuple[np.ma.MaskedArray, tuple[float, float], np.ndarray | None]:
    use_absolute = any(key in config for key in ("limits", "vmin", "vmax", "threshold"))
    layer = get_coregistered_ion_image(
        zarr_path,
        float(config["mz"]),
        msi_dataset=config.get("msi_dataset"),
        ppm_tolerance=float(config.get("ppm_tolerance", ppm_tolerance)),
        reference_key=reference_key,
        normalize_to_tic=bool(config.get("normalize_to_tic", True)),
        mask_low=not use_absolute,
        low_pct=float(config.get("low_pct", 1.0)),
        high_pct=float(config.get("high_pct", 99.8)),
        registered_cs=registered_cs,
        resample_order=int(config.get("resample_order", 0)),
    )
    if not use_absolute:
        return layer.data, layer.contrast_limits, layer.feature_indices

    arr = np.asarray(layer.data, dtype=float)
    if "limits" in config:
        low, high = config["limits"]
        limits = (float(low), float(high))
    else:
        fallback = _display_limits_for_coregistered_image(
            arr,
            low_pct=float(config.get("low_pct", 1.0)),
            high_pct=float(config.get("high_pct", 99.8)),
            positive_only=True,
        )
        low = config.get("threshold", config.get("vmin", fallback[0]))
        high = config.get("vmax", fallback[1])
        limits = (float(low), float(high))
    masked = np.ma.masked_where(~np.isfinite(arr) | (arr <= limits[0]), arr)
    return masked, limits, layer.feature_indices


def export_reference_ion_overlay(
    zarr_path: str | Path,
    ion_layers: Iterable[Mapping[str, Any]],
    *,
    reference_key: str = "hne",
    output_path: str | Path | None = None,
    editable_output_path: str | Path | Iterable[str | Path] | None = None,
    save_editable: bool = False,
    default_msi_dataset: str | None = None,
    ppm_tolerance: float = 5.0,
    normalize_to_tic: bool = True,
    ion_alpha: float = 0.55,
    ion_cmaps: Iterable[str] = ("viridis", "magma", "lime"),
    ion_low_pct: float = 75.0,
    ion_high_pct: float = 99.8,
    ion_resample_order: int = 0,
    reference_channel_overrides: Mapping[Any, Mapping[str, Any]] | None = None,
    reference_blend_mode: str = "translucent",
    respect_saved_if_visibility: bool = True,
    dpi: int = 600,
    figsize: tuple[float, float] | None = None,
    add_colorbars: bool = True,
    colorbar_label_size: float = 8,
    colorbar_tick_size: float = 7,
    colorbar_extend: bool = True,
    colorbar_footprint_fraction: float = 0.16,
    colorbar_height_fraction: float = 0.72,
    colorbar_wspace: float = 0.65,
    scale_bar_length: float | None = None,
    scale_bar_units: str = "um",
    scale_bar_location: str = "lower right",
    scale_bar_color: str | None = None,
    scale_bar_linewidth: float = 3.0,
    scale_bar_fontsize: float = 8.0,
    scale_bar_pad_fraction: float = 0.035,
    scale_bar_label_pad_fraction: float = 0.018,
    scale_bar_box_alpha: float = 0.35,
    reference_pixel_size_um: float | tuple[float, float] | None = None,
    scale_bar_msi_dataset: str | None = None,
    background_color: str = "black",
    foreground_color: str = "white",
    save_pad_inches: float = 0.08,
    show: bool = False,
    save: bool = True,
    registered_cs: str = "registered",
):
    """Export a reference image with one or more coregistered ion overlays.

    TIFF/PNG outputs are standard raster figures. Use `editable_output_path` or
    `save_editable=True` to also save a PDF/SVG-style file where text, scale
    bars, colorbar ticks, and vector patches remain editable in Illustrator.
    """
    zarr_path = Path(zarr_path).expanduser()
    reference_dataset = CoregistrationDataset(zarr_path, registered_cs=registered_cs)
    cmaps = list(ion_cmaps)
    configs = [
        _ion_layer_config(
            layer,
            default_msi_dataset=default_msi_dataset,
            default_cmap=cmaps[idx % len(cmaps)],
            default_alpha=ion_alpha,
            default_low_pct=ion_low_pct,
            default_high_pct=ion_high_pct,
            default_normalize_to_tic=normalize_to_tic,
            default_resample_order=ion_resample_order,
        )
        for idx, layer in enumerate(ion_layers)
    ]
    if not configs:
        raise ValueError("At least one ion layer is required.")

    reference_rgb = reference_rgb_composite(
        reference_dataset,
        reference_key=reference_key,
        channel_overrides=reference_channel_overrides,
        respect_saved_visibility=respect_saved_if_visibility,
        blend_mode=reference_blend_mode,
        registered_cs=registered_cs,
    )
    if output_path is None:
        output_path = zarr_path.with_name(f"{zarr_path.stem}_reference_ion_overlay.tif")
    else:
        output_path = Path(output_path).expanduser()

    if figsize is None:
        height, width = reference_rgb.shape[:2]
        base_width = 8.0
        if add_colorbars:
            footprint = max(0.05, min(float(colorbar_footprint_fraction), 0.6))
            cbar_width = base_width * footprint / (1.0 - footprint)
        else:
            cbar_width = 0.0
        figsize = (base_width + cbar_width, base_width * height / width)

    fig = plt.figure(figsize=figsize, dpi=int(dpi), facecolor=background_color)
    if add_colorbars:
        footprint = max(0.05, min(float(colorbar_footprint_fraction), 0.6))
        cbar_ratio = footprint / (max(len(configs), 1) * (1.0 - footprint))
        grid = gridspec.GridSpec(
            1,
            1 + len(configs),
            figure=fig,
            width_ratios=[1.0] + [cbar_ratio] * len(configs),
            wspace=float(colorbar_wspace),
        )
        ax = fig.add_subplot(grid[0, 0])
        cbar_axes = [fig.add_subplot(grid[0, idx + 1]) for idx in range(len(configs))]
        cbar_height = max(0.2, min(float(colorbar_height_fraction), 1.0))
        if cbar_height < 1.0:
            for cbar_ax in cbar_axes:
                pos = cbar_ax.get_position()
                new_height = pos.height * cbar_height
                y0 = pos.y0 + (pos.height - new_height) / 2.0
                cbar_ax.set_position([pos.x0, y0, pos.width, new_height])
    else:
        ax = fig.add_subplot(111)
        cbar_axes = []

    ax.set_facecolor(background_color)
    ax.imshow(reference_rgb, interpolation="none")
    artists = []
    limits_list = []
    for config in configs:
        ion, limits, feature_indices = _ion_overlay_from_config(
            zarr_path,
            config,
            reference_key=reference_key,
            ppm_tolerance=ppm_tolerance,
            registered_cs=registered_cs,
        )
        artist = ax.imshow(
            ion,
            interpolation="none",
            cmap=_transparent_cmap(str(config["cmap"])),
            vmin=limits[0],
            vmax=limits[1],
            alpha=float(config["alpha"]),
        )
        artists.append(artist)
        limits_list.append(limits)
        feature_count = 0 if feature_indices is None else len(feature_indices)
        print(
            f"Overlay {config.get('msi_dataset')} m/z {float(config['mz']):.4f}: "
            f"{feature_count} feature(s), display limits=({limits[0]:.4g}, {limits[1]:.4g})"
        )

    if scale_bar_length is not None:
        if reference_pixel_size_um is None:
            scale_dataset = reference_dataset
            scale_dataset_selector = scale_bar_msi_dataset or configs[0].get("msi_dataset")
            if scale_dataset_selector:
                table_key, tic_key = _resolve_msi_dataset_keys(
                    zarr_path,
                    msi_dataset=str(scale_dataset_selector),
                )
                scale_dataset = CoregistrationDataset(
                    zarr_path,
                    registered_cs=registered_cs,
                    table_key=table_key,
                    tic_key=tic_key,
                )
            pixel_size_um = _reference_display_pixel_size_from_msi(scale_dataset, reference_key)
            if pixel_size_um is None:
                pixel_size_um = _reference_pixel_size_um(reference_dataset, reference_key)
        elif isinstance(reference_pixel_size_um, (list, tuple)):
            pixel_size_um = (float(reference_pixel_size_um[0]), float(reference_pixel_size_um[1]))
        else:
            pixel_size_um = (float(reference_pixel_size_um), float(reference_pixel_size_um))
        _add_scale_bar(
            ax,
            reference_rgb.shape[:2],
            length=float(scale_bar_length),
            units=str(scale_bar_units),
            pixel_size_um=pixel_size_um,
            location=str(scale_bar_location),
            color=str(scale_bar_color or foreground_color),
            linewidth=float(scale_bar_linewidth),
            fontsize=float(scale_bar_fontsize),
            pad_fraction=float(scale_bar_pad_fraction),
            label_pad_fraction=float(scale_bar_label_pad_fraction),
            box_alpha=float(scale_bar_box_alpha),
        )

    ax.set_axis_off()
    if add_colorbars:
        for cbar_ax, artist, limits, config in zip(cbar_axes, artists, limits_list, configs):
            _add_ion_colorbar(
                cbar_ax,
                artist,
                limits,
                str(config["label"]),
                background_color=background_color,
                foreground_color=foreground_color,
                label_size=colorbar_label_size,
                tick_size=colorbar_tick_size,
                extend=colorbar_extend,
            )
    else:
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    if save:
        _save_figure_for_illustrator(
            fig,
            output_path,
            dpi=int(dpi),
            background_color=background_color,
            save_pad_inches=float(save_pad_inches),
        )
        print(f"Saved {output_path}")
    editable_paths = _iter_output_paths(editable_output_path)
    if save_editable:
        editable_paths.append(Path(output_path).expanduser().with_suffix(".pdf"))
    for editable_path in editable_paths:
        _save_figure_for_illustrator(
            fig,
            editable_path,
            dpi=int(dpi),
            background_color=background_color,
            save_pad_inches=float(save_pad_inches),
        )
        print(f"Saved editable {editable_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def coregistered_campaign_drawgrid(
    root: str | Path,
    *,
    treatment_order: Iterable[str],
    ion_layer: Mapping[str, Any],
    reference_key: str = "hne",
    reference_channel_index: int = 0,
    output_path: str | Path | None = None,
    title: str | None = None,
    msi_dataset: str | None = None,
    ppm_tolerance: float = 5.0,
    normalize_to_tic: bool = True,
    ion_cmap: str = "viridis",
    ion_alpha: float = 0.55,
    ion_low_pct: float = 1.0,
    ion_high_pct: float = 99.8,
    ion_resample_order: int = 0,
    reference_cmap: str = "RdPu",
    reference_alpha: float = 0.3,
    reference_low_pct: float = 10.0,
    reference_high_pct: float = 99.0,
    drawgrid_cut_off: float = 90.0,
    drawgrid_lower_cut_off: float = 1.0,
    group_axis: str = "row",
    dpi: int = 300,
    show: bool = False,
):
    from . import MSI_Process as msi

    root = Path(root).expanduser()
    treatments = list(treatment_order)
    config = _ion_layer_config(
        ion_layer,
        default_msi_dataset=msi_dataset,
        default_cmap=ion_cmap,
        default_alpha=ion_alpha,
        default_low_pct=ion_low_pct,
        default_high_pct=ion_high_pct,
        default_normalize_to_tic=normalize_to_tic,
        default_resample_order=ion_resample_order,
    )

    reference_images = []
    ion_images = []
    names = []
    groups = []
    panel_positions = []

    for treatment_idx, treatment in enumerate(treatments):
        zarr_paths = sorted((root / treatment).glob("*.zarr"), key=lambda path: path.stem.lower())
        for replicate_idx, zarr_path in enumerate(zarr_paths):
            layers = get_coregistered_image_layers(
                zarr_path,
                mzs=[float(config["mz"])],
                msi_dataset=config.get("msi_dataset"),
                reference_key=reference_key,
                reference_channel_index=int(config.get("reference_channel_index", reference_channel_index)),
                ppm_tolerance=float(config.get("ppm_tolerance", ppm_tolerance)),
                normalize_to_tic=bool(config.get("normalize_to_tic", normalize_to_tic)),
                mask_low=True,
                low_pct=float(config.get("low_pct", ion_low_pct)),
                high_pct=float(config.get("high_pct", ion_high_pct)),
                ion_resample_order=int(config.get("resample_order", ion_resample_order)),
            )
            reference_images.append(layers["reference"].data)
            ion_images.append(layers["ions"][0].data)
            names.append(str(config.get("sample_label", zarr_path.stem)) if "sample_label" in config else zarr_path.stem)
            groups.append(treatment)
            panel_positions.append((treatment_idx, replicate_idx))

    if not ion_images:
        raise ValueError(f"No zarr files were found under {root}.")

    max_replicates = max(groups.count(treatment) for treatment in treatments)
    fig = msi.drawGrid(
        ion_images,
        dims=(len(treatments), max_replicates),
        title=title or f"{reference_key} ch {reference_channel_index} with coregistered {config['label']}",
        names=names,
        groups=groups,
        group_axis=group_axis,
        group_order=treatments,
        color_scale="global",
        cut_off=drawgrid_cut_off,
        lower_cut_off=drawgrid_lower_cut_off,
        ignore_zeros=True,
        cmap=_transparent_cmap(str(config.get("cmap", ion_cmap))),
    )

    reference_vmin, reference_vmax = _global_positive_limits(
        reference_images,
        low_pct=float(config.get("reference_low_pct", reference_low_pct)),
        high_pct=float(config.get("reference_high_pct", reference_high_pct)),
    )
    image_axes = np.asarray(fig.axes[: len(treatments) * max_replicates]).reshape(len(treatments), max_replicates)
    for reference_image, _ion_image, (row, column) in zip(reference_images, ion_images, panel_positions):
        ax = image_axes[row, column]
        ion_artist = ax.images[0] if ax.images else None
        if ion_artist is not None:
            ion_artist.set_visible(False)
        ax.imshow(
            _mask_at_or_below(reference_image, reference_vmin),
            interpolation="none",
            cmap=_transparent_cmap(str(config.get("reference_cmap", reference_cmap))),
            vmin=reference_vmin,
            vmax=reference_vmax,
            alpha=float(config.get("reference_alpha", reference_alpha)),
        )
        if ion_artist is not None:
            ion_artist.set_visible(True)
            ion_artist.set_alpha(float(config.get("alpha", ion_alpha)))

    if output_path is not None:
        fig.savefig(Path(output_path).expanduser(), dpi=int(dpi))
        print(f"Saved {output_path}")
    if show:
        plt.show()
    return fig
