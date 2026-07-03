from __future__ import annotations

import json
import tempfile
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
import spatialdata as sd
from spatialdata.models import Image2DModel, ShapesModel
from spatialdata.transformations import (
    Affine,
    Identity,
    get_transformation,
    set_transformation,
)
import imageio.v3 as iio
import geopandas as gpd
import pandas as pd
import tifffile
from matplotlib.path import Path as MplPath
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from shapely import affinity
from spatialdata._io import write_image, write_shapes, write_table


import numpy as np
import zarr
from zarr.errors import ZarrUserWarning

warnings.filterwarnings(
    "ignore",
    message=r"Object at .* is not recognized as a component of a Zarr hierarchy.*",
    category=ZarrUserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Consolidated metadata is currently not part in the Zarr format 3 specification.*",
    category=ZarrUserWarning,
)


def sanitize_name(name: str) -> str:
    safe = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name.strip())
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_").lower()


def _parse_image_to_spatial(img: np.ndarray):
    if img.ndim == 2:
        return Image2DModel.parse(img, dims=("y", "x"))
    if img.ndim == 3:
        return Image2DModel.parse(img, dims=("y", "x", "c"))
    raise ValueError(f"Unexpected image shape: {img.shape}")


def _prepare_qptiff_image(arr: np.ndarray, axes: str) -> tuple[np.ndarray, dict[str, Any]]:
    meta: dict[str, Any] = {"source_axes": axes}
    if axes == "YXS":
        if arr.ndim == 3:
            meta["source_channels"] = int(arr.shape[-1])
        return arr, meta

    if axes in {"SYX", "CYX"}:
        if arr.ndim == 3:
            meta["source_channels"] = int(arr.shape[0])
            return np.moveaxis(arr, 0, -1), meta
        if arr.ndim == 2:
            return arr, meta

    if axes == "YX":
        return arr, meta

    raise ValueError(f"Unsupported qptiff axes {axes!r}")


REFERENCE_CHANNEL_COLOR_PRESETS: dict[str, tuple[float, float, float]] = {
    "metadata": (1.0, 1.0, 1.0),
    "blue": (0.15, 0.4, 1.0),
    "cyan": (0.0, 0.9, 1.0),
    "green": (0.1, 0.95, 0.25),
    "yellow": (1.0, 0.9, 0.1),
    "orange": (1.0, 0.55, 0.0),
    "red": (1.0, 0.2, 0.2),
    "magenta": (0.95, 0.2, 0.95),
    "purple": (0.65, 0.35, 1.0),
    "white": (1.0, 1.0, 1.0),
}


def _extract_qptiff_channel_metadata(tf: tifffile.TiffFile, channel_count: int) -> tuple[list[str], list[tuple[float, float, float]]]:
    names: list[str] = []
    colors: list[tuple[float, float, float]] = []
    for page in tf.pages:
        if len(names) >= channel_count:
            break
        desc = page.description or ""
        if not desc:
            continue
        try:
            root = ET.fromstring(desc)
        except Exception:
            continue
        name_node = root.find("./Name")
        color_node = root.find("./Color")
        if name_node is None:
            continue
        name = (name_node.text or "").strip() or f"ch_{len(names) + 1}"
        color_text = (color_node.text or "").strip() if color_node is not None and color_node.text else ""
        rgb = (1.0, 1.0, 1.0)
        if color_text:
            try:
                parts = [max(0, min(255, int(float(part.strip())))) for part in color_text.split(",")[:3]]
                if len(parts) == 3:
                    rgb = tuple(part / 255.0 for part in parts)
            except Exception:
                pass
        names.append(name)
        colors.append(rgb)
    while len(names) < channel_count:
        names.append(f"ch_{len(names) + 1}")
        colors.append((1.0, 1.0, 1.0))
    return names, colors


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        pass
    try:
        if isinstance(value, (tuple, list)) and len(value) >= 2:
            denom = float(value[1])
            return float(value[0]) / denom if denom else np.nan
    except Exception:
        pass
    try:
        numerator = getattr(value, "numerator")
        denominator = getattr(value, "denominator")
        denominator = float(denominator)
        return float(numerator) / denominator if denominator else np.nan
    except Exception:
        return np.nan


def _unit_to_um(unit: Any) -> float:
    if unit == 2:
        return 25400.0
    if unit == 3:
        return 10000.0
    text = str(unit or "").strip().lower().replace("µ", "u").replace("μ", "u")
    if text in {"um", "micron", "microns", "micrometer", "micrometers", "micrometre", "micrometres"}:
        return 1.0
    if text in {"nm", "nanometer", "nanometers", "nanometre", "nanometres"}:
        return 0.001
    if text in {"mm", "millimeter", "millimeters", "millimetre", "millimetres"}:
        return 1000.0
    if text in {"cm", "centimeter", "centimeters", "centimetre", "centimetres"}:
        return 10000.0
    if text in {"inch", "inches", "in"}:
        return 25400.0
    return np.nan


def _pixel_size_from_description(description: str | None) -> tuple[float, float] | None:
    if not description:
        return None

    def from_mapping(attrs: Mapping[str, Any]) -> tuple[float, float] | None:
        lower = {str(key).lower(): value for key, value in attrs.items()}
        x = _as_float(
            lower.get("physicalsizex")
            or lower.get("pixelsizex")
            or lower.get("pixel_size_x")
            or lower.get("pixelsizemicrons")
            or lower.get("micronsperpixel")
        )
        y = _as_float(
            lower.get("physicalsizey")
            or lower.get("pixelsizey")
            or lower.get("pixel_size_y")
            or lower.get("pixelsizemicrons")
            or lower.get("micronsperpixel")
        )
        x_unit = lower.get("physicalsizexunit") or lower.get("pixelsizexunit") or lower.get("unit") or "um"
        y_unit = lower.get("physicalsizeyunit") or lower.get("pixelsizeyunit") or lower.get("unit") or x_unit
        x_factor = _unit_to_um(x_unit)
        y_factor = _unit_to_um(y_unit)
        if np.isfinite(x) and x > 0 and np.isfinite(y) and y > 0:
            if not np.isfinite(x_factor):
                x_factor = 1.0
            if not np.isfinite(y_factor):
                y_factor = x_factor
            return float(x * x_factor), float(y * y_factor)
        return None

    try:
        root = ET.fromstring(description)
        for elem in root.iter():
            result = from_mapping(elem.attrib)
            if result is not None:
                return result
            text = (elem.text or "").strip()
            tag = str(elem.tag).split("}")[-1].lower()
            if text and tag in {"pixelsizemicrons", "micronsperpixel"}:
                value = _as_float(text)
                if np.isfinite(value) and value > 0:
                    return float(value), float(value)
    except Exception:
        pass
    return None


def _pixel_size_from_tiff_page(page) -> tuple[float, float] | None:
    tags = getattr(page, "tags", {})
    description = None
    try:
        description = page.description
    except Exception:
        pass
    result = _pixel_size_from_description(description)
    if result is not None:
        return result

    try:
        x_res = _as_float(tags["XResolution"].value)
        y_res = _as_float(tags["YResolution"].value)
    except Exception:
        return None
    if not np.isfinite(x_res) or x_res <= 0 or not np.isfinite(y_res) or y_res <= 0:
        return None
    try:
        resolution_unit = tags["ResolutionUnit"].value
    except Exception:
        resolution_unit = ""
    unit_um = _unit_to_um(resolution_unit)
    if not np.isfinite(unit_um):
        try:
            resolution_unit = tags["ResolutionUnit"].value.name
            unit_um = _unit_to_um(resolution_unit)
        except Exception:
            pass
    if not np.isfinite(unit_um):
        return None
    return float(unit_um / x_res), float(unit_um / y_res)


def _read_tiff_fullres_pixel_size_um(path: Path) -> tuple[float, float] | None:
    try:
        with tifffile.TiffFile(path) as tf:
            pages = []
            try:
                series = tf.series[0]
                levels = list(getattr(series, "levels", []) or [series])
                pages.append(levels[0].pages[0])
            except Exception:
                pass
            try:
                pages.append(tf.pages[0])
            except Exception:
                pass
            for page in pages:
                result = _pixel_size_from_tiff_page(page)
                if result is not None:
                    return result
    except Exception:
        return None
    return None


def _annotation_scale_from_pyramid_level(
    target_attrs: Mapping[str, Any] | dict[str, Any] | Any,
    annotation_pyramid_level: int | None,
) -> tuple[float | None, float | None]:
    if annotation_pyramid_level is None or int(annotation_pyramid_level) < 0 or not isinstance(target_attrs, Mapping):
        return None, None
    requested_level = int(annotation_pyramid_level)
    current_scale_x = target_attrs.get("fullres_to_image_scale_x")
    current_scale_y = target_attrs.get("fullres_to_image_scale_y")
    current_level = target_attrs.get("qptiff_level")
    try:
        if current_scale_x is not None and current_scale_y is not None and current_level is not None:
            level_delta = int(current_level) - requested_level
            factor = float(2**level_delta)
            return float(current_scale_x) * factor, float(current_scale_y) * factor
    except Exception:
        pass
    try:
        fallback = float(1.0 / (2**requested_level))
    except Exception:
        return None, None
    return fallback, fallback


def _read_qptiff_image(path: Path, *, level: int = 0) -> tuple[np.ndarray, dict[str, Any]]:
    with tifffile.TiffFile(path) as tf:
        if not tf.series:
            raise ValueError(f"No TIFF series found in {path}")
        series = tf.series[0]
        levels = list(getattr(series, "levels", []) or [series])
        if level < 0 or level >= len(levels):
            raise ValueError(f"Requested qptiff level {level} not available; found 0..{len(levels) - 1}")
        full_level = levels[0]
        selected = levels[level]
        arr = selected.asarray()
        axes = getattr(selected, "axes", getattr(series, "axes", ""))
        img, display_meta = _prepare_qptiff_image(arr, axes)
        channel_count = int(display_meta.get("source_channels", 0)) if isinstance(display_meta, dict) else 0
        channel_names, channel_colors = _extract_qptiff_channel_metadata(tf, channel_count) if channel_count else ([], [])

        full_shape = tuple(getattr(full_level, "shape", img.shape))
        selected_shape = tuple(getattr(selected, "shape", img.shape))
        if "X" not in axes or "Y" not in axes:
            raise ValueError(f"Unsupported qptiff axes {axes!r} for {path}")
        x_idx = axes.index("X")
        y_idx = axes.index("Y")
        fullres_to_image_scale_x = float(selected_shape[x_idx]) / float(full_shape[x_idx])
        fullres_to_image_scale_y = float(selected_shape[y_idx]) / float(full_shape[y_idx])
        image_to_fullres_scale_x = float(full_shape[x_idx]) / float(selected_shape[x_idx])
        image_to_fullres_scale_y = float(full_shape[y_idx]) / float(selected_shape[y_idx])
        meta = {
            "qptiff_level": int(level),
            "fullres_to_image_scale_x": fullres_to_image_scale_x,
            "fullres_to_image_scale_y": fullres_to_image_scale_y,
            "image_to_fullres_scale_x": image_to_fullres_scale_x,
            "image_to_fullres_scale_y": image_to_fullres_scale_y,
            "image_source": "qptiff_pyramid",
        }
        meta.update(display_meta)
        if channel_names:
            meta["channel_names"] = channel_names
            meta["channel_colors"] = channel_colors
        return img, meta


def _clone_spatial_image_element(image) -> Any:
    data = np.asarray(image).copy()
    dims = tuple(getattr(image, "dims", ()))
    if dims and len(dims) == data.ndim:
        return Image2DModel.parse(data, dims=dims)
    return _parse_image_to_spatial(data)


def _write_element_to_existing_store(
    zarr_path: str | Path,
    *,
    element: Any,
    element_type: str,
    element_name: str,
    overwrite: bool = True,
    consolidate_metadata: bool = True,
) -> None:
    """Write a new element into an existing SpatialData store without mutating a backed object in-place."""
    target_path = Path(zarr_path).expanduser()
    if overwrite:
        try:
            root = zarr.open_group(target_path, mode="r+", use_consolidated=False)
            if element_type in root and element_name in root[element_type]:
                del root[element_type][element_name]
        except Exception:
            pass
    scratch = sd.SpatialData()
    scratch._write_element(  # type: ignore[attr-defined]
        element=element,
        zarr_container_path=target_path,
        element_type=element_type,
        element_name=element_name,
        overwrite=overwrite,
    )
    if consolidate_metadata:
        sd.read_zarr(target_path).write_consolidated_metadata()


def _geojson_json_compatible(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"nan", "null", "none"}:
            return None
        return value
    if isinstance(value, Mapping):
        return {str(k): _geojson_json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_geojson_json_compatible(v) for v in value]
    return str(value)


def _normalize_geojson_property_value(value: Any) -> Any:
    normalized = _geojson_json_compatible(value)
    if isinstance(normalized, (dict, list)):
        return json.dumps(normalized, ensure_ascii=True, sort_keys=isinstance(normalized, dict))
    return normalized


def _extract_annotation_label(row: Mapping[str, Any]) -> str:
    for key in ("name", "classification", "classification_name", "label", "description"):
        if key not in row:
            continue
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, Mapping):
            nested_name = value.get("name")
            if nested_name is not None and str(nested_name).strip():
                return str(nested_name).strip()
        text = str(value).strip()
        if not text:
            continue
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, Mapping):
                nested_name = parsed.get("name")
                if nested_name is not None and str(nested_name).strip():
                    return str(nested_name).strip()
        return text
    return ""


def _sanitize_geojson_annotations(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    sanitized = gdf.copy()
    geometry_name = sanitized.geometry.name
    if "_annotation_label" not in sanitized.columns:
        sanitized["_annotation_label"] = [
            _extract_annotation_label({col: sanitized.iloc[idx][col] for col in sanitized.columns if col != geometry_name})
            for idx in range(len(sanitized))
        ]
    for col in sanitized.columns:
        if col == geometry_name:
            continue
        series = sanitized[col]
        if series.dtype != object:
            continue
        sanitized[col] = series.map(_normalize_geojson_property_value)
        non_null_types = {type(v) for v in sanitized[col] if v is not None}
        if len(non_null_types) <= 1:
            continue
        if non_null_types.issubset({int, float, bool}):
            sanitized[col] = sanitized[col].map(lambda v: None if v is None else float(v))
        else:
            sanitized[col] = sanitized[col].map(lambda v: None if v is None else str(v))
    return sanitized


def _filter_and_downsample_geojson_annotations(
    gdf: gpd.GeoDataFrame,
    *,
    object_mode: str = "all",
    max_shapes: int = 0,
    simplify_tolerance: float = 0.0,
) -> gpd.GeoDataFrame:
    filtered = gdf
    if "objectType" in filtered.columns:
        object_types = filtered["objectType"].fillna("").astype(str).str.strip().str.lower()
        if object_mode == "annotations_only":
            filtered = filtered.loc[object_types == "annotation"].copy()
        elif object_mode == "non_cell":
            filtered = filtered.loc[object_types != "cell"].copy()
        elif object_mode == "cells_only":
            filtered = filtered.loc[object_types == "cell"].copy()

    if max_shapes and max_shapes > 0 and len(filtered) > int(max_shapes):
        keep_idx = np.linspace(0, len(filtered) - 1, int(max_shapes), dtype=int)
        filtered = filtered.iloc[keep_idx].copy()

    if simplify_tolerance and simplify_tolerance > 0:
        filtered = filtered.copy()
        filtered.geometry = filtered.geometry.simplify(float(simplify_tolerance), preserve_topology=True)
        filtered = filtered.loc[~filtered.geometry.is_empty].copy()

    return filtered


def _scale_geojson_annotations(
    gdf: gpd.GeoDataFrame,
    *,
    annotation_scale: float = 1.0,
    annotation_scale_x: float | None = None,
    annotation_scale_y: float | None = None,
) -> gpd.GeoDataFrame:
    sx = float(annotation_scale_x) if annotation_scale_x is not None else float(annotation_scale)
    sy = float(annotation_scale_y) if annotation_scale_y is not None else float(annotation_scale)
    if np.isclose(sx, 1.0) and np.isclose(sy, 1.0):
        return gdf
    scaled = gdf.copy()
    scaled.geometry = scaled.geometry.map(
        lambda geom: None if geom is None or geom.is_empty else affinity.scale(geom, xfact=sx, yfact=sy, origin=(0.0, 0.0))
    )
    return scaled.loc[~scaled.geometry.is_empty].copy()


def _transform_geojson_annotations(
    gdf: gpd.GeoDataFrame,
    *,
    annotation_scale: float = 1.0,
    annotation_scale_x: float | None = None,
    annotation_scale_y: float | None = None,
    annotation_translate_x: float = 0.0,
    annotation_translate_y: float = 0.0,
) -> gpd.GeoDataFrame:
    transformed = _scale_geojson_annotations(
        gdf,
        annotation_scale=annotation_scale,
        annotation_scale_x=annotation_scale_x,
        annotation_scale_y=annotation_scale_y,
    )
    tx = float(annotation_translate_x)
    ty = float(annotation_translate_y)
    if np.isclose(tx, 0.0) and np.isclose(ty, 0.0):
        return transformed
    shifted = transformed.copy()
    shifted.geometry = shifted.geometry.map(
        lambda geom: None if geom is None or geom.is_empty else affinity.translate(geom, xoff=tx, yoff=ty)
    )
    return shifted.loc[~shifted.geometry.is_empty].copy()


def _to_napari_image(arr: np.ndarray):
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim == 3 and arr.shape[0] in (3, 4):
        return np.moveaxis(arr, 0, -1)
    return arr


def _make_reference_channel_colormap(rgb: tuple[float, float, float], name: str):
    from napari.utils.colormaps import Colormap

    r, g, b = rgb
    colors = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [r * 0.35, g * 0.35, b * 0.35, 0.6],
            [r, g, b, 1.0],
        ],
        dtype=float,
    )
    return Colormap(colors=colors, name=name)


def _fallback_reference_channel_color(index: int) -> tuple[float, float, float]:
    palette = [
        (0.95, 0.25, 0.25),  # red
        (0.20, 0.80, 0.30),  # green
        (0.20, 0.45, 0.95),  # blue
        (0.95, 0.75, 0.20),  # yellow
        (0.85, 0.30, 0.90),  # magenta
        (0.10, 0.85, 0.85),  # cyan
        (1.00, 0.55, 0.15),  # orange
        (0.70, 0.85, 0.20),  # lime
    ]
    return palette[index % len(palette)]


def _extract_xy(vec, axes):
    vals = np.asarray(vec, dtype=float).ravel()
    if axes is not None:
        axes = tuple(axes)
        if "x" in axes and "y" in axes:
            return float(vals[axes.index("x")]), float(vals[axes.index("y")])
    if vals.size == 2:
        return float(vals[0]), float(vals[1])
    if vals.size >= 3:
        return float(vals[-1]), float(vals[-2])
    raise RuntimeError(f"Cannot extract x/y from vector with shape {vals.shape}")


def _xy_matrix_from_transform(tr) -> np.ndarray:
    name = tr.__class__.__name__
    if name == "Identity":
        return np.eye(3, dtype=float)
    if name == "Scale":
        vec = getattr(tr, "vector", None)
        if vec is None:
            vec = getattr(tr, "scale", None)
        sx, sy = _extract_xy(vec, getattr(tr, "axes", None))
        return np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    if name == "Translation":
        vec = getattr(tr, "vector", None)
        if vec is None:
            vec = getattr(tr, "translation", None)
        tx, ty = _extract_xy(vec, getattr(tr, "axes", None))
        return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=float)
    if name == "Affine":
        m = np.asarray(getattr(tr, "matrix"), dtype=float)
        in_axes = tuple(getattr(tr, "input_axes"))
        out_axes = tuple(getattr(tr, "output_axes"))
        in_x, in_y = in_axes.index("x"), in_axes.index("y")
        out_x, out_y = out_axes.index("x"), out_axes.index("y")
        tcol = m.shape[1] - 1
        return np.array(
            [
                [m[out_x, in_x], m[out_x, in_y], m[out_x, tcol]],
                [m[out_y, in_x], m[out_y, in_y], m[out_y, tcol]],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
    if name == "Sequence":
        transforms = getattr(tr, "transformations", None)
        if transforms is None:
            transforms = getattr(tr, "_transformations", None)
        if transforms is None:
            raise RuntimeError(f"Couldn't inspect Sequence transform: {tr}")
        total = np.eye(3, dtype=float)
        for item in transforms:
            total = _xy_matrix_from_transform(item) @ total
        return total
    raise RuntimeError(f"Unsupported transform type for auto-load: {name}")


def xy_to_yx_matrix(matrix_xy: np.ndarray) -> np.ndarray:
    swap = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return swap @ np.asarray(matrix_xy, dtype=float) @ swap


def auto_contrast_limits(img: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.5) -> tuple[float, float]:
    if np.ma.isMaskedArray(img):
        finite = np.asarray(img.compressed(), dtype=float)
    else:
        vals = np.asarray(img, dtype=float)
        finite = vals[np.isfinite(vals)]
        finite = finite[finite > 0]
    if finite.size == 0:
        return (0.0, 1.0)
    lo = float(np.percentile(finite, low_pct))
    hi = float(np.percentile(finite, high_pct))
    if hi <= lo:
        hi = lo + 1e-9
    return (lo, hi)


def finite_data_limits(img: np.ndarray) -> tuple[float, float]:
    if np.ma.isMaskedArray(img):
        finite = np.asarray(img.compressed(), dtype=float)
    else:
        vals = np.asarray(img, dtype=float)
        finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return (0.0, 1.0)
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi <= lo:
        hi = lo + 1e-9
    return (lo, hi)


def normalize_image_for_registration(img: np.ndarray) -> np.ndarray:
    data = np.asarray(img, dtype=float).copy()
    finite = np.isfinite(data)
    if not np.any(finite):
        return np.zeros(data.shape, dtype=np.float32)
    fill_value = float(np.nanmin(data[finite]))
    data[~finite] = fill_value
    finite_values = data[np.isfinite(data)]
    lo = float(np.percentile(finite_values, 1.0))
    hi = float(np.percentile(finite_values, 99.5))
    if hi <= lo:
        hi = lo + 1e-9
    data = np.clip(data, lo, hi)
    data = (data - lo) / (hi - lo)
    return data.astype(np.float32, copy=False)


def sitk_affine_from_fixed_to_moving_matrix(matrix_xy: np.ndarray):
    import SimpleITK as sitk

    matrix = np.asarray(matrix_xy, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("SITK initializer matrix must be a finite 3x3 affine matrix.")
    transform = sitk.AffineTransform(2)
    transform.SetCenter((0.0, 0.0))
    transform.SetMatrix(tuple(matrix[:2, :2].ravel()))
    transform.SetTranslation(tuple(matrix[:2, 2]))
    return transform


def sitk_transform_to_homogeneous_matrix(transform) -> np.ndarray:
    import SimpleITK as sitk

    if isinstance(transform, sitk.CompositeTransform):
        if transform.GetNumberOfTransforms() != 1:
            raise ValueError("Composite transform has multiple components.")
        transform = transform.GetNthTransform(0)
    if transform.GetDimension() != 2:
        raise ValueError("Only 2D transforms are supported.")
    if transform.GetName() != "AffineTransform":
        transform = sitk.AffineTransform(transform)

    params = list(transform.GetParameters())
    fixed_params = list(transform.GetFixedParameters())
    a00, a01, a10, a11, tx, ty = params
    cx, cy = fixed_params
    linear = np.array([[a00, a01], [a10, a11]], dtype=float)
    translation = np.array([tx, ty], dtype=float)
    center = np.array([cx, cy], dtype=float)
    offset = center + translation - linear @ center

    matrix = np.eye(3, dtype=float)
    matrix[:2, :2] = linear
    matrix[:2, 2] = offset
    return matrix


def prepare_ion_for_display(img: np.ndarray) -> np.ndarray:
    data = np.asarray(img, dtype=float).copy()
    data[(~np.isfinite(data)) | (data <= 0)] = 0.0
    return data


def _infer_default_zarr_path(input_path: str | Path) -> Path:
    src = Path(input_path).expanduser()
    return src.with_suffix(".zarr")


def convert_input_to_zarr(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    converter: Any | None = None,
) -> Path:
    src = Path(input_path).expanduser()
    dst = Path(output_path).expanduser() if output_path is not None else _infer_default_zarr_path(src)
    if dst.suffix.lower() != ".zarr":
        dst = dst.with_suffix(".zarr")

    if converter is None:
        try:
            from thyra import convert_msi as converter
        except ImportError as exc:
            raise ImportError(
                "Converting `.imzML`/`.npz` into SpatialData zarr currently requires "
                "`thyra.convert_msi` to be installed."
            ) from exc

    ok = converter(input_path=src, output_path=str(dst))
    if ok is False:
        ok = converter(input_path=src, output_path=str(dst),pixel_size_um=50)
        if ok is False:
            raise RuntimeError(f"Failed to convert MSI input {src} -> {dst}")
    return dst


def _sanitize_dataset_label(value: str | Path) -> str:
    if isinstance(value, Path):
        value = value.stem or value.name
    return sanitize_name(str(value)) or "msi"


def _infer_msi_dataset_specs(sdata) -> list[dict[str, Any]]:
    tic_keys = [key for key in sdata.images.keys() if key.endswith("_tic")]
    tic_set = set(tic_keys)
    table_keys = list(sdata.tables.keys())
    specs = []
    used_tics: set[str] = set()

    for idx, table_key in enumerate(table_keys):
        table = sdata.tables[table_key]
        uns = getattr(table, "uns", {}) or {}
        label = _sanitize_dataset_label(uns.get("coregistration_dataset_label") or table_key)
        stored_tic = uns.get("coregistration_tic_key")

        tic_key = None
        if isinstance(stored_tic, str) and stored_tic in sdata.images:
            tic_key = stored_tic
        elif f"{table_key}_tic" in sdata.images:
            tic_key = f"{table_key}_tic"
        elif f"{label}_tic" in sdata.images:
            tic_key = f"{label}_tic"
        elif len(table_keys) == 1 and len(tic_keys) == 1:
            tic_key = tic_keys[0]
        elif idx < len(tic_keys):
            candidate = tic_keys[idx]
            if candidate not in used_tics:
                tic_key = candidate

        if tic_key is None:
            continue
        used_tics.add(tic_key)

        raw_pixel_shape_keys = uns.get("coregistration_pixel_shape_keys", [])
        pixel_shape_keys = [key for key in raw_pixel_shape_keys if isinstance(key, str) and key in sdata.shapes]
        if not pixel_shape_keys and len(table_keys) == 1:
            pixel_shape_keys = [key for key in sdata.shapes.keys() if "pixels" in key.lower()]

        specs.append(
            {
                "label": label,
                "display_name": str(uns.get("coregistration_display_name") or label),
                "table_key": table_key,
                "tic_key": tic_key,
                "pixel_shape_keys": pixel_shape_keys,
            }
        )

    return specs


def list_coregistration_msi_datasets(zarr_path: str | Path) -> list[dict[str, Any]]:
    sdata = sd.read_zarr(Path(zarr_path).expanduser())
    return [dict(spec) for spec in _infer_msi_dataset_specs(sdata)]


def _resolve_msi_dataset_keys(
    zarr_path: str | Path,
    *,
    msi_dataset: str | None = None,
    table_key: str | None = None,
    tic_key: str | None = None,
) -> tuple[str | None, str | None]:
    if not msi_dataset:
        return table_key, tic_key

    sdata = sd.read_zarr(Path(zarr_path).expanduser())
    specs = _infer_msi_dataset_specs(sdata)
    query = str(msi_dataset).strip()
    query_folded = query.casefold()
    query_sanitized = sanitize_name(query)

    def selector_values(spec: Mapping[str, Any]) -> list[str]:
        return [
            str(spec.get("display_name", "")),
            str(spec.get("label", "")),
            str(spec.get("table_key", "")),
            str(spec.get("tic_key", "")),
        ]

    def matches_at_rank(rank: int) -> list[dict[str, Any]]:
        matches = []
        for spec in specs:
            raw_values = selector_values(spec)
            folded_values = [value.casefold() for value in raw_values]
            safe_values = [sanitize_name(value) for value in raw_values]
            if rank == 0 and query in raw_values:
                matches.append(spec)
            elif rank == 1 and query_folded in folded_values:
                matches.append(spec)
            elif rank == 2 and query_sanitized in safe_values:
                matches.append(spec)
            elif rank == 3 and query_folded and any(query_folded in value for value in folded_values):
                matches.append(spec)
            elif rank == 4 and query_sanitized and any(query_sanitized in value for value in safe_values):
                matches.append(spec)
        return matches

    matches: list[dict[str, Any]] = []
    for rank in range(5):
        matches = matches_at_rank(rank)
        if matches:
            break

    if not matches:
        available = ", ".join(str(spec.get("display_name") or spec.get("table_key")) for spec in specs)
        raise ValueError(f"No MSI dataset matched {msi_dataset!r}. Available datasets: {available}")
    if len(matches) > 1:
        labels = ", ".join(str(spec.get("display_name") or spec.get("table_key")) for spec in matches)
        raise ValueError(f"MSI dataset selector {msi_dataset!r} matched multiple datasets: {labels}")
    selected = matches[0]
    return str(selected["table_key"]), str(selected["tic_key"])


def _choose_unique_element_key(existing: set[str], base: str) -> str:
    candidate = _sanitize_dataset_label(base)
    if candidate not in existing:
        return candidate
    suffix = 2
    while f"{candidate}_{suffix}" in existing:
        suffix += 1
    return f"{candidate}_{suffix}"


def _describe_zarr_layout(zarr_path: str | Path) -> dict[str, Any]:
    root = zarr.open_group(str(Path(zarr_path).expanduser()), mode="r")
    layout: dict[str, Any] = {"top_level_keys": list(root.keys())}
    for key in ("images", "labels", "shapes", "points", "tables"):
        if key in root:
            try:
                layout[key] = list(root[key].keys())
            except Exception:
                layout[key] = []
    return layout


def embed_msi_dataset(
    host_zarr_path: str | Path,
    source_path: str | Path,
    *,
    dataset_label: str | None = None,
    registered_cs: str = "registered",
    converter: Any | None = None,
) -> dict[str, Any]:

    host_zarr = Path(host_zarr_path).expanduser()
    src = Path(source_path).expanduser()

    cleanup_dir: tempfile.TemporaryDirectory[str] | None = None
    if src.suffix.lower() == ".zarr" and src.exists():
        source_zarr = src
    else:
        cleanup_dir = tempfile.TemporaryDirectory(prefix="viu_chem_coreg_")
        source_zarr = convert_input_to_zarr(src, Path(cleanup_dir.name) / f"{src.stem}.zarr", converter=converter)

    try:
        host_sdata = sd.read_zarr(host_zarr)
        try:
            source_sdata = sd.read_zarr(source_zarr)
        except Exception as exc:
            layout = _describe_zarr_layout(source_zarr)
            if not layout.get("tables"):
                raise ValueError(
                    "Selected .zarr is not a compatible MSI dataset for `Add MSI Dataset`. "
                    f"It does not contain any SpatialData tables and appears to be image-only or partial. "
                    f"Top-level keys: {layout.get('top_level_keys', [])}. "
                    "Use an MSI `.imzML`, `.npz`, or a full SpatialData MSI `.zarr` that includes both `tables/` "
                    "and the MSI TIC image."
                ) from exc
            raise ValueError(
                "Failed to read the selected .zarr as a compatible SpatialData MSI dataset for `Add MSI Dataset`. "
                f"Layout summary: {layout}"
            ) from exc
        source_specs = _infer_msi_dataset_specs(source_sdata)
        if not source_specs:
            raise ValueError(f"No MSI dataset found in source zarr: {source_zarr}")
        source_spec = source_specs[0]

        requested_label = dataset_label or source_spec["label"] or src.stem
        existing_keys = set(host_sdata.tables.keys()) | set(host_sdata.images.keys()) | set(host_sdata.shapes.keys())
        label = _choose_unique_element_key(existing_keys, requested_label)
        table_key = label
        tic_key = f"{label}_tic"

        table = source_sdata.tables[source_spec["table_key"]].copy()
        table.uns["coregistration_dataset_label"] = label
        table.uns["coregistration_display_name"] = str(dataset_label or source_spec.get("display_name") or label)
        table.uns["coregistration_tic_key"] = tic_key

        source_tic = source_sdata.images[source_spec["tic_key"]]
        tic_element = _clone_spatial_image_element(source_tic)
        for attr_key, attr_value in getattr(source_tic, "attrs", {}).items():
            tic_element.attrs[attr_key] = attr_value

        transforms: dict[str, Any] = {}
        for cs in ("global", registered_cs):
            try:
                transforms[cs] = get_transformation(source_sdata.images[source_spec["tic_key"]], to_coordinate_system=cs)
            except Exception:
                pass
        if "global" not in transforms:
            transforms["global"] = Identity()
        if registered_cs not in transforms:
            transforms[registered_cs] = Identity()
        for cs, transform in transforms.items():
            set_transformation(tic_element, transform, to_coordinate_system=cs)

        pixel_shape_keys = []
        shape_elements: list[tuple[str, Any]] = []
        for old_key in source_spec["pixel_shape_keys"]:
            new_key = _choose_unique_element_key(existing_keys | {table_key, tic_key, *pixel_shape_keys}, f"{label}_{old_key}")
            shape_element = source_sdata.shapes[old_key].copy()
            for cs, transform in transforms.items():
                try:
                    set_transformation(shape_element, transform, to_coordinate_system=cs)
                except Exception:
                    pass
            shape_elements.append((new_key, shape_element))
            pixel_shape_keys.append(new_key)

        table.uns["coregistration_pixel_shape_keys"] = list(pixel_shape_keys)
        root = zarr.open_group(str(host_zarr), mode="a", use_consolidated=False)
        write_table(table, root.require_group("tables"), table_key)
        write_image(tic_element, root.require_group("images").require_group(tic_key), tic_key)
        shapes_root = root.require_group("shapes")
        for new_key, shape_element in shape_elements:
            write_shapes(shape_element, shapes_root.require_group(new_key))
        zarr.consolidate_metadata(str(host_zarr))
        return {
            "label": label,
            "table_key": table_key,
            "tic_key": tic_key,
            "pixel_shape_keys": pixel_shape_keys,
        }
    finally:
        if cleanup_dir is not None:
            cleanup_dir.cleanup()


def rename_msi_dataset(
    zarr_path: str | Path,
    *,
    table_key: str,
    display_name: str,
) -> str:
    from spatialdata._io import write_table

    zarr_path = Path(zarr_path).expanduser()
    sdata = sd.read_zarr(zarr_path)
    if table_key not in sdata.tables:
        raise KeyError(f"MSI dataset table not found: {table_key}")

    cleaned = str(display_name).strip() or table_key
    table = sdata.tables[table_key].copy()
    table.uns["coregistration_display_name"] = cleaned
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    write_table(table, root.require_group("tables"), table_key)
    zarr.consolidate_metadata(str(zarr_path))
    return cleaned


def delete_msi_dataset(
    zarr_path: str | Path,
    *,
    table_key: str,
) -> dict[str, list[str]]:
    host_zarr_path = Path(zarr_path).expanduser()
    sdata = sd.read_zarr(host_zarr_path)
    specs = _infer_msi_dataset_specs(sdata)
    selected = next((spec for spec in specs if str(spec["table_key"]) == str(table_key)), None)
    if selected is None:
        raise KeyError(f"MSI dataset table not found: {table_key}")

    root = zarr.open_group(str(host_zarr_path), mode="a", use_consolidated=False)
    deleted: dict[str, list[str]] = {"tables": [], "images": [], "shapes": []}
    element_keys = {
        "tables": [str(selected["table_key"])],
        "images": [str(selected["tic_key"])],
        "shapes": [str(key) for key in selected.get("pixel_shape_keys", [])],
    }

    for element_type, keys in element_keys.items():
        if element_type not in root:
            continue
        group = root[element_type]
        for key in keys:
            if key not in group:
                continue
            del group[key]
            deleted[element_type].append(key)

    if any(deleted.values()):
        zarr.consolidate_metadata(str(host_zarr_path))
    return deleted


@dataclass
class CoregistrationDataset:
    zarr_path: Path
    registered_cs: str = "registered"
    table_key: str | None = None
    tic_key: str | None = None

    def __post_init__(self) -> None:
        self.zarr_path = Path(self.zarr_path).expanduser()
        self.sdata = sd.read_zarr(self.zarr_path)

        specs = _infer_msi_dataset_specs(self.sdata)
        if not specs:
            raise ValueError(f"No MSI dataset found in {self.zarr_path}")

        selected = None
        for spec in specs:
            if self.table_key is not None and spec["table_key"] == self.table_key:
                selected = spec
                break
            if self.tic_key is not None and spec["tic_key"] == self.tic_key:
                selected = spec
                break
        if selected is None:
            selected = specs[0]

        self.table_key = selected["table_key"]
        self.tic_key = selected["tic_key"]
        self.dataset_label = selected["label"]
        self.display_name = selected.get("display_name", self.dataset_label)
        self.pixel_shape_keys = list(selected["pixel_shape_keys"])
        self.msi_table = self.sdata.tables[self.table_key]
        self.mz_values = self.msi_table.var["mz"].values.astype(float)
        self.X = self.msi_table.X
        self.tic_array = np.asarray(self.sdata.images[self.tic_key])[0]
        all_tic_keys = {spec["tic_key"] for spec in specs}
        self.reference_image_keys = [key for key in self.sdata.images.keys() if key not in all_tic_keys]

        self.x_coords = self.msi_table.obs["x"].values.astype(int)
        self.y_coords = self.msi_table.obs["y"].values.astype(int)
        self.ny, self.nx = self.tic_array.shape
        self.pixel_tic_values = self.tic_array[self.y_coords, self.x_coords].astype(float)

        avg_spectrum = self.msi_table.uns.get("average_spectrum")
        if avg_spectrum is None:
            avg_spectrum = np.asarray(self.X.mean(axis=0)).ravel()
        else:
            avg_spectrum = np.asarray(avg_spectrum, dtype=float).ravel()
            if avg_spectrum.shape[0] != self.mz_values.shape[0]:
                avg_spectrum = np.asarray(self.X.mean(axis=0)).ravel()
        self.avg_spectrum = avg_spectrum

        self.local_maxima_indices = self._find_local_maxima()

    def _find_local_maxima(self) -> np.ndarray:
        mask = np.zeros_like(self.avg_spectrum, dtype=bool)
        if self.avg_spectrum.size >= 3:
            mask[1:-1] = (
                (self.avg_spectrum[1:-1] >= self.avg_spectrum[:-2])
                & (self.avg_spectrum[1:-1] > self.avg_spectrum[2:])
            )
        idx = np.flatnonzero(mask)
        return idx if idx.size else np.arange(self.avg_spectrum.size)

    def reconstruct_ion_image(self, feature_idx: int | Iterable[int] | np.ndarray, *, normalize_to_tic: bool = True) -> np.ndarray:
        feature_indices = np.atleast_1d(np.asarray(feature_idx, dtype=int))
        if feature_indices.size == 1:
            col = self.X[:, int(feature_indices[0])]
            ion_values = np.asarray(col.toarray()).ravel() if hasattr(col, "toarray") else np.asarray(col).ravel()
        else:
            cols = self.X[:, feature_indices]
            ion_values = np.asarray(cols.sum(axis=1)).ravel()
        img = np.zeros((self.ny, self.nx), dtype=float)
        img[self.y_coords, self.x_coords] = ion_values
        if normalize_to_tic:
            with np.errstate(divide="ignore", invalid="ignore"):
                img = np.divide(
                    img,
                    self.tic_array,
                    out=np.zeros_like(img, dtype=float),
                    where=self.tic_array != 0,
                )
        return img

    def find_feature_idx_from_mz(self, target_mz: float, ppm_tolerance: float = 5.0) -> tuple[int | None, float]:
        if target_mz <= 0:
            return None, float("inf")
        ppm_errors = np.abs(self.mz_values - target_mz) / target_mz * 1e6
        idx = int(np.argmin(ppm_errors))
        ppm_error = float(ppm_errors[idx])
        if ppm_error <= ppm_tolerance:
            return idx, ppm_error
        return None, ppm_error

    def find_feature_indices_from_mz(self, target_mz: float, ppm_tolerance: float = 5.0) -> np.ndarray:
        if target_mz <= 0 or ppm_tolerance <= 0:
            return np.array([], dtype=int)
        ppm_errors = np.abs(self.mz_values - target_mz) / target_mz * 1e6
        return np.flatnonzero(ppm_errors <= ppm_tolerance).astype(int)

    def find_local_max_idx_near_mz(self, target_mz: float, mz_window: tuple[float, float] | None = None) -> int:
        candidate_indices = self.local_maxima_indices
        if mz_window is not None:
            lo, hi = float(min(mz_window)), float(max(mz_window))
            in_window = candidate_indices[(self.mz_values[candidate_indices] >= lo) & (self.mz_values[candidate_indices] <= hi)]
            if in_window.size:
                strongest_local = int(np.argmax(self.avg_spectrum[in_window]))
                return int(in_window[strongest_local])
        local_idx = int(np.argmin(np.abs(self.mz_values[candidate_indices] - target_mz)))
        return int(candidate_indices[local_idx])

    def summarize_region_spectra(
        self,
        selected_mask: np.ndarray,
        *,
        normalize_to_tic: bool = True,
    ) -> dict[str, np.ndarray | int]:
        selected = np.asarray(selected_mask, dtype=bool).ravel()
        if selected.shape[0] != self.x_coords.shape[0]:
            raise ValueError("Selected mask must match the number of MSI spectra.")
        selected_idx = np.flatnonzero(selected)
        n_spectra = int(selected_idx.size)
        if n_spectra == 0:
            raise ValueError("No MSI spectra were selected for export.")
        subset = self.X[selected_idx, :]
        dense = np.asarray(subset.toarray() if hasattr(subset, "toarray") else subset, dtype=float)
        if normalize_to_tic:
            tic = self.pixel_tic_values[selected_idx]
            with np.errstate(divide="ignore", invalid="ignore"):
                dense = np.divide(
                    dense,
                    tic[:, None],
                    out=np.zeros_like(dense, dtype=float),
                    where=tic[:, None] != 0,
                )
        return {
            "mz": self.mz_values.copy(),
            "mean_intensity": dense.mean(axis=0),
            "std_intensity": dense.std(axis=0, ddof=0),
            "n_spectra": n_spectra,
        }

    def load_saved_registration_if_available(self) -> tuple[np.ndarray, bool]:
        try:
            transform = get_transformation(
                self.sdata.images[self.tic_key],
                to_coordinate_system=self.registered_cs,
            )
        except Exception:
            return np.eye(3, dtype=float), False
        try:
            return _xy_matrix_from_transform(transform), True
        except Exception:
            return np.eye(3, dtype=float), False

    def estimate_initial_scale_from_pixel_sizes(self) -> tuple[np.ndarray, tuple[Any, ...] | None]:
        def as_pos_float(value: Any) -> float:
            try:
                result = float(value)
            except Exception:
                return np.nan
            return result if np.isfinite(result) and result > 0 else np.nan

        raw = self.msi_table.uns.get("raw_metadata", {})
        attrs = getattr(self.sdata, "attrs", {})
        detection = attrs.get("pixel_size_detection_info", {})

        raw_x = as_pos_float(raw.get("pixel size x")) if isinstance(raw, dict) else np.nan
        raw_y = as_pos_float(raw.get("pixel size y")) if isinstance(raw, dict) else np.nan
        if np.isfinite(raw_x) and np.isfinite(raw_y):
            msi_um_x, msi_um_y, source = raw_x, raw_y, "table.uns['raw_metadata']"
        else:
            det_x = as_pos_float(detection.get("detected_x_um")) if isinstance(detection, dict) else np.nan
            det_y = as_pos_float(detection.get("detected_y_um")) if isinstance(detection, dict) else np.nan
            if np.isfinite(det_x) and np.isfinite(det_y):
                msi_um_x, msi_um_y, source = det_x, det_y, "sdata.attrs['pixel_size_detection_info']"
            else:
                attr_x = as_pos_float(attrs.get("pixel_size_x_um"))
                attr_y = as_pos_float(attrs.get("pixel_size_y_um"))
                if np.isfinite(attr_x) and np.isfinite(attr_y):
                    msi_um_x, msi_um_y, source = attr_x, attr_y, "sdata.attrs"
                else:
                    return np.eye(3, dtype=float), None

        if not self.reference_image_keys:
            # No reference pixel size is available yet, but the MSI pixel
            # dimensions still tell us the aspect ratio. Normalize by the
            # smaller dimension so square MSI pixels remain identity and a
            # later reference image can still provide the absolute scale.
            base_um = min(msi_um_x, msi_um_y)
            if not np.isfinite(base_um) or base_um <= 0:
                return np.eye(3, dtype=float), None
            sx = msi_um_x / base_um
            sy = msi_um_y / base_um
            return np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=float), (
                None,
                msi_um_x,
                msi_um_y,
                base_um,
                base_um,
                source,
            )

        preferred = [key for key in ("hne", "optical") if key in self.reference_image_keys]
        ref_key = preferred[0] if preferred else self.reference_image_keys[0]
        ref_attrs = getattr(self.sdata.images[ref_key], "attrs", {})
        ref_um_x = as_pos_float(ref_attrs.get("pixel_size_x_um"))
        ref_um_y = as_pos_float(ref_attrs.get("pixel_size_y_um"))
        if not np.isfinite(ref_um_x):
            ref_um_x = 2.54
        if not np.isfinite(ref_um_y):
            ref_um_y = 2.54

        sx = msi_um_x / ref_um_x
        sy = msi_um_y / ref_um_y
        return np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=float), (
            ref_key,
            msi_um_x,
            msi_um_y,
            ref_um_x,
            ref_um_y,
            source,
        )

    def registration_sidecar_path(self) -> Path:
        return self.zarr_path.with_name(f"{self.zarr_path.name}.{self.dataset_label}.coregistration_params.json")


def add_reference_image(
    zarr_path: str | Path,
    image_path: str | Path,
    *,
    key: str,
    registered_cs: str = "registered",
    qptiff_level: int | None = None,
) -> CoregistrationDataset:
    if key not in {"optical", "hne"}:
        raise ValueError("Reference image `key` must be either 'optical' or 'hne'.")

    host_zarr_path = Path(zarr_path).expanduser()
    dataset = CoregistrationDataset(host_zarr_path, registered_cs=registered_cs)
    saved_display_settings: dict[str, Any] = {}
    try:
        root = zarr.open_group(host_zarr_path, mode="r", use_consolidated=False)
        if "images" in root and key in root["images"]:
            raw_settings = root["images"][key].attrs.get("if_display_settings", {})
            if isinstance(raw_settings, Mapping):
                saved_display_settings = dict(raw_settings)
    except Exception:
        pass
    source_path = Path(image_path).expanduser()
    qptiff_meta: dict[str, float | int | str] = {}
    if source_path.suffix.lower() in {".qptiff", ".ome.tiff", ".ome.tif"} and qptiff_level is not None:
        img, qptiff_meta = _read_qptiff_image(source_path, level=int(qptiff_level))
    else:
        img = iio.imread(source_path)
    element = _parse_image_to_spatial(img)

    px_um_x = 2.54
    px_um_y = 2.54
    pixel_size_source = "fallback_10000dpi"
    tiff_pixel_size = _read_tiff_fullres_pixel_size_um(source_path)
    if tiff_pixel_size is not None:
        px_um_x, px_um_y = tiff_pixel_size
        pixel_size_source = "tifffile_metadata"
    else:
        try:
            meta = iio.immeta(source_path)
            dpi = meta.get("dpi")
            if dpi is not None:
                if isinstance(dpi, (tuple, list)) and len(dpi) >= 2:
                    dx, dy = float(dpi[0]), float(dpi[1])
                else:
                    dx = dy = float(dpi)
                if dx > 0:
                    px_um_x = 25400.0 / dx
                    pixel_size_source = "imageio_dpi"
                if dy > 0:
                    px_um_y = 25400.0 / dy
                    pixel_size_source = "imageio_dpi"
        except Exception:
            pass

    if qptiff_meta:
        px_um_x *= float(qptiff_meta["image_to_fullres_scale_x"])
        px_um_y *= float(qptiff_meta["image_to_fullres_scale_y"])

    element.attrs["pixel_size_x_um"] = float(px_um_x)
    element.attrs["pixel_size_y_um"] = float(px_um_y)
    element.attrs["pixel_size_source"] = pixel_size_source
    element.attrs["source_path"] = str(source_path)
    if saved_display_settings:
        element.attrs["if_display_settings"] = saved_display_settings
    for attr_key, attr_value in qptiff_meta.items():
        element.attrs[attr_key] = attr_value

    set_transformation(element, Identity(), to_coordinate_system="global")
    set_transformation(element, Identity(), to_coordinate_system=registered_cs)
    _write_element_to_existing_store(
        host_zarr_path,
        element=element,
        element_type="images",
        element_name=key,
        overwrite=True,
        consolidate_metadata=True,
    )
    dataset.sdata = sd.read_zarr(host_zarr_path)
    return dataset


def import_geojson_annotations(
    zarr_path: str | Path,
    annotation_paths: Iterable[str | Path],
    *,
    target_image: str = "hne",
    name_prefix: str = "anno_",
    registered_cs: str = "registered",
    object_mode: str = "all",
    max_shapes: int = 0,
    simplify_tolerance: float = 0.0,
    annotation_scale: float = 1.0,
    annotation_pyramid_level: int | None = None,
    annotation_scale_x: float | None = None,
    annotation_scale_y: float | None = None,
    annotation_translate_x: float = 0.0,
    annotation_translate_y: float = 0.0,
) -> list[str]:

    host_zarr_path = Path(zarr_path).expanduser()
    dataset = CoregistrationDataset(host_zarr_path, registered_cs=registered_cs)
    target = target_image.strip() or "hne"
    if target not in dataset.sdata.images:
        if dataset.reference_image_keys:
            target = dataset.reference_image_keys[0]
        else:
            raise ValueError("No reference image available to anchor annotations.")

    target_transforms: dict[str, Any] = {}
    for cs in ("global", registered_cs):
        try:
            target_transforms[cs] = get_transformation(dataset.sdata.images[target], to_coordinate_system=cs)
        except Exception:
            pass

    target_attrs = getattr(dataset.sdata.images[target], "attrs", {})
    if annotation_pyramid_level is not None and int(annotation_pyramid_level) >= 0:
        level_scale_x, level_scale_y = _annotation_scale_from_pyramid_level(target_attrs, annotation_pyramid_level)
        if level_scale_x is not None:
            annotation_scale_x = float(level_scale_x)
        if level_scale_y is not None:
            annotation_scale_y = float(level_scale_y)
        annotation_scale = 1.0
    if (
        annotation_scale_x is None
        and annotation_scale_y is None
        and np.isclose(float(annotation_scale), 1.0)
        and isinstance(target_attrs, Mapping)
    ):
        auto_scale_x = target_attrs.get("fullres_to_image_scale_x")
        auto_scale_y = target_attrs.get("fullres_to_image_scale_y")
        try:
            if auto_scale_x is not None:
                annotation_scale_x = float(auto_scale_x)
            if auto_scale_y is not None:
                annotation_scale_y = float(auto_scale_y)
        except Exception:
            annotation_scale_x = None
            annotation_scale_y = None

    imported = []
    for annotation_path in annotation_paths:
        src = Path(annotation_path).expanduser()
        gdf = _sanitize_geojson_annotations(gpd.read_file(src))
        gdf = _filter_and_downsample_geojson_annotations(
            gdf,
            object_mode=object_mode,
            max_shapes=max_shapes,
            simplify_tolerance=simplify_tolerance,
        )
        gdf = _transform_geojson_annotations(
            gdf,
            annotation_scale=annotation_scale,
            annotation_scale_x=annotation_scale_x,
            annotation_scale_y=annotation_scale_y,
            annotation_translate_x=annotation_translate_x,
            annotation_translate_y=annotation_translate_y,
        )
        if gdf.empty:
            continue
        key = f"{name_prefix}{sanitize_name(src.stem)}"
        shape_element = ShapesModel.parse(gdf)
        for cs, transform in target_transforms.items():
            set_transformation(
                shape_element,
                transform,
                to_coordinate_system=cs,
            )
        _write_element_to_existing_store(
            host_zarr_path,
            element=shape_element,
            element_type="shapes",
            element_name=key,
            overwrite=True,
            consolidate_metadata=False,
        )
        imported.append(key)

    if imported:
        sd.read_zarr(host_zarr_path).write_consolidated_metadata()
    return imported


def delete_geojson_annotations(
    zarr_path: str | Path,
    annotation_keys: Iterable[str],
) -> list[str]:
    host_zarr_path = Path(zarr_path).expanduser()
    keys = [str(key) for key in annotation_keys]
    if not keys:
        return []
    try:
        sdata = sd.read_zarr(host_zarr_path)
    except Exception:
        sdata = None
    root = zarr.open_group(str(host_zarr_path), mode="a", use_consolidated=False)
    shapes_root = root.require_group("shapes")
    deleted: list[str] = []
    for key in keys:
        if sdata is not None and key in sdata.shapes:
            try:
                sdata.delete_element_from_disk(key)
                deleted.append(key)
                continue
            except Exception:
                pass
        if key in shapes_root:
            del shapes_root[key]
            deleted.append(key)
    if deleted:
        zarr.consolidate_metadata(str(host_zarr_path))
    return deleted


def rescale_geojson_annotations(
    zarr_path: str | Path,
    annotation_keys: Iterable[str],
    *,
    annotation_scale: float = 1.0,
    annotation_scale_x: float | None = None,
    annotation_scale_y: float | None = None,
) -> list[str]:
    host_zarr_path = Path(zarr_path).expanduser()
    keys = [str(key) for key in annotation_keys]
    if not keys:
        return []
    sdata = sd.read_zarr(host_zarr_path)
    rewritten: list[str] = []
    for key in keys:
        if key not in sdata.shapes:
            continue
        gdf = sdata.shapes[key]
        scaled = _scale_geojson_annotations(
            gdf,
            annotation_scale=annotation_scale,
            annotation_scale_x=annotation_scale_x,
            annotation_scale_y=annotation_scale_y,
        )
        transforms = get_transformation(gdf, get_all=True)
        sdata.delete_element_from_disk(key)
        shape_element = ShapesModel.parse(scaled)
        set_transformation(shape_element, transforms, set_all=True)
        _write_element_to_existing_store(
            host_zarr_path,
            element=shape_element,
            element_type="shapes",
            element_name=key,
            overwrite=True,
            consolidate_metadata=False,
        )
        rewritten.append(key)
    if rewritten:
        sd.read_zarr(host_zarr_path).write_consolidated_metadata()
    return rewritten


def transform_geojson_annotations(
    zarr_path: str | Path,
    annotation_keys: Iterable[str],
    *,
    annotation_scale: float = 1.0,
    annotation_scale_x: float | None = None,
    annotation_scale_y: float | None = None,
    annotation_translate_x: float = 0.0,
    annotation_translate_y: float = 0.0,
) -> list[str]:
    host_zarr_path = Path(zarr_path).expanduser()
    keys = [str(key) for key in annotation_keys]
    if not keys:
        return []
    sdata = sd.read_zarr(host_zarr_path)
    rewritten: list[str] = []
    for key in keys:
        if key not in sdata.shapes:
            continue
        gdf = sdata.shapes[key]
        transformed = _transform_geojson_annotations(
            gdf,
            annotation_scale=annotation_scale,
            annotation_scale_x=annotation_scale_x,
            annotation_scale_y=annotation_scale_y,
            annotation_translate_x=annotation_translate_x,
            annotation_translate_y=annotation_translate_y,
        )
        transforms = get_transformation(gdf, get_all=True)
        sdata.delete_element_from_disk(key)
        shape_element = ShapesModel.parse(transformed)
        set_transformation(shape_element, transforms, set_all=True)
        _write_element_to_existing_store(
            host_zarr_path,
            element=shape_element,
            element_type="shapes",
            element_name=key,
            overwrite=True,
            consolidate_metadata=False,
        )
        rewritten.append(key)
    if rewritten:
        sd.read_zarr(host_zarr_path).write_consolidated_metadata()
    return rewritten


@dataclass
class MSIThresholdMask:
    target_mz: float
    ppm_tolerance: float
    feature_indices: np.ndarray
    actual_mzs: np.ndarray
    threshold: float
    mode: str
    normalize_to_tic: bool
    values: np.ndarray
    allowed_mask: np.ndarray
    below_mask: np.ndarray
    above_mask: np.ndarray
    below_image: np.ndarray
    above_image: np.ndarray


@dataclass
class MSIThresholdSpectra:
    mask: MSIThresholdMask
    below_summary: dict[str, np.ndarray | int] | None
    above_summary: dict[str, np.ndarray | int] | None


def create_msi_threshold_mask(
    zarr_path: str | Path | CoregistrationDataset,
    *,
    target_mz: float,
    ppm_tolerance: float = 5.0,
    threshold: float | None = None,
    percentile: float | None = None,
    normalize_to_tic: bool = True,
    msi_dataset: str | None = None,
    table_key: str | None = None,
    tic_key: str | None = None,
    prefilter_mask: np.ndarray | None = None,
    registered_cs: str = "registered",
) -> MSIThresholdMask:
    if (threshold is None) == (percentile is None):
        raise ValueError("Supply exactly one of `threshold` or `percentile`.")

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

    indices = dataset.find_feature_indices_from_mz(float(target_mz), float(ppm_tolerance))
    if indices.size == 0:
        idx, _ = dataset.find_feature_idx_from_mz(float(target_mz), float("inf"))
        if idx is None:
            raise ValueError(f"No m/z feature found near {target_mz:g}.")
        indices = np.array([idx], dtype=int)

    img = dataset.reconstruct_ion_image(indices, normalize_to_tic=bool(normalize_to_tic))
    values = img[dataset.y_coords, dataset.x_coords]
    allowed = np.isfinite(values)
    if prefilter_mask is not None:
        prefilter = np.asarray(prefilter_mask, dtype=bool).ravel()
        if prefilter.shape[0] != values.shape[0]:
            raise ValueError("prefilter_mask must match the number of MSI spectra.")
        allowed &= prefilter
    if not np.any(allowed):
        raise ValueError("No MSI pixels have finite intensities after filtering.")

    if percentile is not None:
        pct = float(percentile)
        if not 0 <= pct <= 100:
            raise ValueError("percentile must be between 0 and 100.")
        threshold_value = float(np.percentile(values[allowed], pct))
        mode = "percentile"
    else:
        threshold_value = float(threshold)
        if not np.isfinite(threshold_value):
            raise ValueError("threshold must be finite.")
        mode = "absolute"

    below = (values < threshold_value) & allowed
    above = (values >= threshold_value) & allowed
    if not np.any(below) and not np.any(above):
        raise ValueError("Threshold selected no MSI pixels.")

    below_image = np.zeros((dataset.ny, dataset.nx), dtype=np.uint8)
    above_image = np.zeros((dataset.ny, dataset.nx), dtype=np.uint8)
    below_image[dataset.y_coords[below], dataset.x_coords[below]] = 1
    above_image[dataset.y_coords[above], dataset.x_coords[above]] = 1

    return MSIThresholdMask(
        target_mz=float(target_mz),
        ppm_tolerance=float(ppm_tolerance),
        feature_indices=indices.astype(int),
        actual_mzs=dataset.mz_values[indices].copy(),
        threshold=threshold_value,
        mode=mode,
        normalize_to_tic=bool(normalize_to_tic),
        values=values,
        allowed_mask=allowed,
        below_mask=below,
        above_mask=above,
        below_image=below_image,
        above_image=above_image,
    )


def summarize_msi_threshold_spectra(
    zarr_path: str | Path | CoregistrationDataset,
    *,
    target_mz: float,
    ppm_tolerance: float = 5.0,
    threshold: float | None = None,
    percentile: float | None = None,
    threshold_normalize_to_tic: bool = True,
    summary_normalize_to_tic: bool = True,
    msi_dataset: str | None = None,
    table_key: str | None = None,
    tic_key: str | None = None,
    prefilter_mask: np.ndarray | None = None,
    registered_cs: str = "registered",
) -> MSIThresholdSpectra:
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
    mask = create_msi_threshold_mask(
        dataset,
        target_mz=target_mz,
        ppm_tolerance=ppm_tolerance,
        threshold=threshold,
        percentile=percentile,
        normalize_to_tic=threshold_normalize_to_tic,
        prefilter_mask=prefilter_mask,
    )
    below_summary = (
        dataset.summarize_region_spectra(mask.below_mask, normalize_to_tic=bool(summary_normalize_to_tic))
        if np.any(mask.below_mask)
        else None
    )
    above_summary = (
        dataset.summarize_region_spectra(mask.above_mask, normalize_to_tic=bool(summary_normalize_to_tic))
        if np.any(mask.above_mask)
        else None
    )
    return MSIThresholdSpectra(
        mask=mask,
        below_summary=below_summary,
        above_summary=above_summary,
    )


def _colocalization_threshold(values: np.ndarray, threshold: str | float | int) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    if isinstance(threshold, str):
        mode = threshold.strip().lower()
        if mode == "median":
            return float(np.median(finite))
        if mode == "mean":
            return float(np.mean(finite))
        if mode.startswith("p"):
            return float(np.percentile(finite, float(mode[1:])))
        raise ValueError("threshold must be 'median', 'mean', 'pNN', or a percentile number.")
    pct = float(threshold)
    if not 0 <= pct <= 100:
        raise ValueError("Numeric threshold must be a percentile between 0 and 100.")
    return float(np.percentile(finite, pct))


def _colocalized_features_for_values(
    dataset: CoregistrationDataset,
    reference_values: np.ndarray,
    *,
    n: int | None = 10,
    sort_by: str = "correlation",
    threshold: str | float = "median",
    normalize_to_tic: bool = True,
    prefilter_mask: np.ndarray | None = None,
    chunk_size: int = 512,
    metadata: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    selected = np.ones(len(dataset.x_coords), dtype=bool)
    reference_values = np.asarray(reference_values, dtype=float).ravel()
    if reference_values.shape[0] != selected.shape[0]:
        raise ValueError("reference_values must match the number of target MSI spectra.")
    selected &= np.isfinite(reference_values)
    if prefilter_mask is not None:
        prefilter = np.asarray(prefilter_mask, dtype=bool).ravel()
        if prefilter.shape[0] != selected.shape[0]:
            raise ValueError("prefilter_mask must match the number of MSI spectra.")
        selected &= prefilter
    selected_idx = np.flatnonzero(selected)
    if selected_idx.size < 2:
        raise ValueError("At least two MSI pixels are required for colocalization.")

    ref = reference_values[selected]
    ref = np.where(np.isfinite(ref), ref, 0.0)
    ref_mask_threshold = _colocalization_threshold(ref, threshold)
    ref_mask = ref > ref_mask_threshold
    ref_count = int(np.count_nonzero(ref_mask))
    ref_centered = ref - float(np.mean(ref))
    ref_norm = float(np.linalg.norm(ref_centered))

    sort_key = str(sort_by).strip()
    allowed_sort = {"correlation", "M1", "M2", "abs_correlation", "anti_M1", "anti_M2", "anti_score"}
    if sort_key not in allowed_sort:
        raise ValueError(f"sort_by must be one of {sorted(allowed_sort)}.")

    rows: list[dict[str, Any]] = []
    n_features = int(dataset.mz_values.shape[0])
    chunk_size = max(1, int(chunk_size))
    tic = np.asarray(dataset.pixel_tic_values[selected_idx], dtype=float)

    for start in range(0, n_features, chunk_size):
        stop = min(n_features, start + chunk_size)
        subset = dataset.X[selected_idx, start:stop]
        dense = np.asarray(subset.toarray() if hasattr(subset, "toarray") else subset, dtype=float)
        dense[~np.isfinite(dense)] = 0.0
        if normalize_to_tic:
            with np.errstate(divide="ignore", invalid="ignore"):
                dense = np.divide(
                    dense,
                    tic[:, None],
                    out=np.zeros_like(dense, dtype=float),
                    where=tic[:, None] != 0,
                )

        positive_counts = np.count_nonzero(dense > 0, axis=0).astype(float)
        positive_fraction = positive_counts / float(selected_idx.size)
        signal_sums = np.sum(dense, axis=0)
        signal_sums_sq = np.sum(dense * dense, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            participation_ratio = np.divide(
                signal_sums * signal_sums,
                signal_sums_sq,
                out=np.zeros_like(signal_sums, dtype=float),
                where=signal_sums_sq > 0,
            )
        effective_fraction = participation_ratio / float(selected_idx.size)

        means = np.mean(dense, axis=0)
        centered = dense - means
        norms = np.linalg.norm(centered, axis=0)
        if ref_norm > 0:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                correlations = (ref_centered @ centered) / (ref_norm * norms)
        else:
            correlations = np.full(stop - start, np.nan, dtype=float)
        correlations = np.asarray(correlations, dtype=float)
        correlations[~np.isfinite(correlations)] = np.nan

        feature_thresholds = np.apply_along_axis(_colocalization_threshold, 0, dense, threshold)
        feature_masks = dense > feature_thresholds[None, :]
        overlap = np.count_nonzero(feature_masks & ref_mask[:, None], axis=0).astype(float)
        anti_ref = np.count_nonzero((~feature_masks) & ref_mask[:, None], axis=0).astype(float)
        anti_feature = np.count_nonzero(feature_masks & (~ref_mask[:, None]), axis=0).astype(float)
        feature_counts = np.count_nonzero(feature_masks, axis=0).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            m1 = np.divide(overlap, ref_count, out=np.full_like(overlap, np.nan, dtype=float), where=ref_count > 0)
            m2 = np.divide(overlap, feature_counts, out=np.full_like(overlap, np.nan, dtype=float), where=feature_counts > 0)
            anti_m1 = np.divide(anti_ref, ref_count, out=np.full_like(anti_ref, np.nan, dtype=float), where=ref_count > 0)
            anti_m2 = np.divide(anti_feature, feature_counts, out=np.full_like(anti_feature, np.nan, dtype=float), where=feature_counts > 0)
            anti_score = np.divide(
                2.0 * anti_m1 * anti_m2,
                anti_m1 + anti_m2,
                out=np.full_like(anti_m1, np.nan, dtype=float),
                where=np.isfinite(anti_m1) & np.isfinite(anti_m2) & ((anti_m1 + anti_m2) > 0),
            )

        for offset, feature_idx in enumerate(range(start, stop)):
            corr = float(correlations[offset])
            rows.append(
                {
                    "feature_index": int(feature_idx),
                    "mz": float(dataset.mz_values[feature_idx]),
                    "correlation": corr,
                    "abs_correlation": abs(corr) if np.isfinite(corr) else np.nan,
                    "M1": float(m1[offset]),
                    "M2": float(m2[offset]),
                    "anti_M1": float(anti_m1[offset]),
                    "anti_M2": float(anti_m2[offset]),
                    "anti_score": float(anti_score[offset]),
                    "n_overlap": int(overlap[offset]),
                    "n_reference_high_feature_low": int(anti_ref[offset]),
                    "n_reference_low_feature_high": int(anti_feature[offset]),
                    "n_reference_mask": ref_count,
                    "n_feature_mask": int(feature_counts[offset]),
                    "feature_high_fraction": float(feature_counts[offset] / float(selected_idx.size)),
                    "positive_fraction": float(positive_fraction[offset]),
                    "effective_fraction": float(effective_fraction[offset]),
                    "sparsity": float(1.0 - positive_fraction[offset]),
                    "effective_sparsity": float(1.0 - effective_fraction[offset]),
                    "mean_intensity": float(means[offset]),
                    "total_intensity": float(signal_sums[offset]),
                    "normalize_to_tic": bool(normalize_to_tic),
                    "target_dataset": str(dataset.display_name),
                }
            )
            if metadata:
                rows[-1].update(metadata)

    out = pd.DataFrame(rows)
    out = out.sort_values(sort_key, ascending=False, na_position="last").reset_index(drop=True)
    if n is not None and np.isfinite(float(n)):
        out = out.head(int(n)).reset_index(drop=True)
    return out


def colocalized_msi_features(
    zarr_path: str | Path | CoregistrationDataset,
    *,
    reference_mz: float,
    ppm_tolerance: float = 5.0,
    n: int | None = 10,
    sort_by: str = "correlation",
    threshold: str | float = "median",
    normalize_to_tic: bool = True,
    msi_dataset: str | None = None,
    table_key: str | None = None,
    tic_key: str | None = None,
    prefilter_mask: np.ndarray | None = None,
    chunk_size: int = 512,
    registered_cs: str = "registered",
) -> pd.DataFrame:
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

    ref_indices = dataset.find_feature_indices_from_mz(float(reference_mz), float(ppm_tolerance))
    if ref_indices.size == 0:
        idx, _ = dataset.find_feature_idx_from_mz(float(reference_mz), float("inf"))
        if idx is None:
            raise ValueError(f"No m/z feature found near {reference_mz:g}.")
        ref_indices = np.array([idx], dtype=int)

    ref_img = dataset.reconstruct_ion_image(ref_indices, normalize_to_tic=bool(normalize_to_tic))
    ref_values = np.asarray(ref_img[dataset.y_coords, dataset.x_coords], dtype=float)
    return _colocalized_features_for_values(
        dataset,
        ref_values,
        n=n,
        sort_by=sort_by,
        threshold=threshold,
        normalize_to_tic=normalize_to_tic,
        prefilter_mask=prefilter_mask,
        chunk_size=chunk_size,
        metadata={
            "reference_mz": float(reference_mz),
            "reference_actual_mzs": ",".join(f"{mz:.8g}" for mz in dataset.mz_values[ref_indices]),
            "ppm_tolerance": float(ppm_tolerance),
            "reference_dataset": str(dataset.display_name),
        },
    )


def colocalized_msi_features_between_datasets(
    zarr_path: str | Path,
    *,
    source_msi_dataset: str | None = None,
    target_msi_dataset: str | None = None,
    source_table_key: str | None = None,
    source_tic_key: str | None = None,
    target_table_key: str | None = None,
    target_tic_key: str | None = None,
    reference_mz: float,
    ppm_tolerance: float = 5.0,
    n: int | None = 10,
    sort_by: str = "correlation",
    threshold: str | float = "median",
    source_normalize_to_tic: bool = True,
    target_normalize_to_tic: bool = True,
    source_transform_xy: np.ndarray | None = None,
    target_transform_xy: np.ndarray | None = None,
    prefilter_mask: np.ndarray | None = None,
    chunk_size: int = 512,
    registered_cs: str = "registered",
) -> pd.DataFrame:
    source_table_key, source_tic_key = _resolve_msi_dataset_keys(
        zarr_path,
        msi_dataset=source_msi_dataset,
        table_key=source_table_key,
        tic_key=source_tic_key,
    )
    target_table_key, target_tic_key = _resolve_msi_dataset_keys(
        zarr_path,
        msi_dataset=target_msi_dataset,
        table_key=target_table_key,
        tic_key=target_tic_key,
    )
    source_dataset = CoregistrationDataset(
        zarr_path,
        registered_cs=registered_cs,
        table_key=source_table_key,
        tic_key=source_tic_key,
    )
    target_dataset = CoregistrationDataset(
        zarr_path,
        registered_cs=registered_cs,
        table_key=target_table_key,
        tic_key=target_tic_key,
    )

    ref_indices = source_dataset.find_feature_indices_from_mz(float(reference_mz), float(ppm_tolerance))
    if ref_indices.size == 0:
        idx, _ = source_dataset.find_feature_idx_from_mz(float(reference_mz), float("inf"))
        if idx is None:
            raise ValueError(f"No m/z feature found near {reference_mz:g} in source MSI dataset.")
        ref_indices = np.array([idx], dtype=int)

    source_img = source_dataset.reconstruct_ion_image(ref_indices, normalize_to_tic=bool(source_normalize_to_tic))
    if source_transform_xy is None:
        source_transform_xy, _found = source_dataset.load_saved_registration_if_available()
    if target_transform_xy is None:
        target_transform_xy, _found = target_dataset.load_saved_registration_if_available()
    reference_values = _sample_msi_values_at_msi_pixels(
        source_img,
        source_dataset,
        np.asarray(source_transform_xy, dtype=float),
        target_dataset,
        np.asarray(target_transform_xy, dtype=float),
    )

    return _colocalized_features_for_values(
        target_dataset,
        reference_values,
        n=n,
        sort_by=sort_by,
        threshold=threshold,
        normalize_to_tic=target_normalize_to_tic,
        prefilter_mask=prefilter_mask,
        chunk_size=chunk_size,
        metadata={
            "reference_mz": float(reference_mz),
            "reference_actual_mzs": ",".join(f"{mz:.8g}" for mz in source_dataset.mz_values[ref_indices]),
            "ppm_tolerance": float(ppm_tolerance),
            "reference_dataset": str(source_dataset.display_name),
            "source_normalize_to_tic": bool(source_normalize_to_tic),
            "target_normalize_to_tic": bool(target_normalize_to_tic),
        },
    )


def correlate_msi_features_with_reference_channels(
    zarr_path: str | Path | CoregistrationDataset,
    *,
    reference_key: str = "hne",
    channel_indices: Iterable[int] | None = None,
    n: int | None = 10,
    sort_by: str = "correlation",
    threshold: str | float = "median",
    normalize_to_tic: bool = True,
    msi_dataset: str | None = None,
    table_key: str | None = None,
    tic_key: str | None = None,
    prefilter_mask: np.ndarray | None = None,
    transform_xy: np.ndarray | None = None,
    chunk_size: int = 512,
    registered_cs: str = "registered",
) -> pd.DataFrame:
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
    if transform_xy is None:
        transform_xy, _found = dataset.load_saved_registration_if_available()
    channel_count, channel_names = _reference_channel_count_and_names(dataset.sdata, reference_key)
    channels = list(range(channel_count)) if channel_indices is None else [int(idx) for idx in channel_indices]

    tables = []
    for channel_index in channels:
        if channel_index < 0 or channel_index >= channel_count:
            raise ValueError(f"Channel {channel_index} is outside reference image {reference_key!r}.")
        reference_img = _reference_channel_image(dataset.sdata, reference_key, channel_index)
        reference_values = _sample_reference_values_at_msi_pixels(reference_img, dataset, np.asarray(transform_xy, dtype=float))
        table = _colocalized_features_for_values(
            dataset,
            reference_values,
            n=n,
            sort_by=sort_by,
            threshold=threshold,
            normalize_to_tic=normalize_to_tic,
            prefilter_mask=prefilter_mask,
            chunk_size=chunk_size,
            metadata={
                "reference_key": str(reference_key),
                "reference_channel_index": int(channel_index),
                "reference_channel_name": channel_names[channel_index],
                "msi_dataset": str(dataset.display_name),
            },
        )
        tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def summarize_reference_channels_in_msi_mask(
    zarr_path: str | Path | CoregistrationDataset,
    pixel_mask: np.ndarray,
    *,
    reference_key: str = "hne",
    channel_indices: Iterable[int] | None = None,
    msi_dataset: str | None = None,
    table_key: str | None = None,
    tic_key: str | None = None,
    transform_xy: np.ndarray | None = None,
    registered_cs: str = "registered",
) -> pd.DataFrame:
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
    selected = np.asarray(pixel_mask, dtype=bool).ravel()
    if selected.shape[0] != len(dataset.x_coords):
        raise ValueError("pixel_mask must match the number of MSI spectra.")
    if not np.any(selected):
        raise ValueError("pixel_mask selected no MSI pixels.")
    if transform_xy is None:
        transform_xy, _found = dataset.load_saved_registration_if_available()
    channel_count, channel_names = _reference_channel_count_and_names(dataset.sdata, reference_key)
    channels = list(range(channel_count)) if channel_indices is None else [int(idx) for idx in channel_indices]

    rows = []
    for channel_index in channels:
        if channel_index < 0 or channel_index >= channel_count:
            raise ValueError(f"Channel {channel_index} is outside reference image {reference_key!r}.")
        reference_img = _reference_channel_image(dataset.sdata, reference_key, channel_index)
        values = _sample_reference_values_at_msi_pixels(reference_img, dataset, np.asarray(transform_xy, dtype=float))
        selected_values = values[selected]
        finite = selected_values[np.isfinite(selected_values)]
        rows.append(
            {
                "reference_key": str(reference_key),
                "channel_index": int(channel_index),
                "channel_name": channel_names[channel_index],
                "n_pixels": int(np.count_nonzero(selected)),
                "n_finite": int(finite.size),
                "mean_intensity": float(np.mean(finite)) if finite.size else np.nan,
                "median_intensity": float(np.median(finite)) if finite.size else np.nan,
                "std_intensity": float(np.std(finite, ddof=0)) if finite.size else np.nan,
                "min_intensity": float(np.min(finite)) if finite.size else np.nan,
                "max_intensity": float(np.max(finite)) if finite.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def create_msi_threshold_annotation(
    zarr_path: str | Path,
    *,
    table_key: str | None = None,
    tic_key: str | None = None,
    target_mz: float,
    ppm_tolerance: float = 5.0,
    threshold: float,
    normalize_to_tic: bool = True,
    transform_xy: np.ndarray | None = None,
    prefilter_mask: np.ndarray | None = None,
    prefilter_shape_key: str = "",
    prefilter_region_label: str = "",
    annotation_name: str = "",
    annotation_label: str = "",
    registered_cs: str = "registered",
) -> str:
    host_zarr_path = Path(zarr_path).expanduser()
    dataset = CoregistrationDataset(host_zarr_path, registered_cs=registered_cs, table_key=table_key, tic_key=tic_key)
    indices = dataset.find_feature_indices_from_mz(float(target_mz), float(ppm_tolerance))
    if indices.size == 0:
        idx, _ = dataset.find_feature_idx_from_mz(float(target_mz), float("inf"))
        if idx is None:
            raise ValueError(f"No m/z feature found near {target_mz:g}.")
        indices = np.array([idx], dtype=int)

    img = dataset.reconstruct_ion_image(indices, normalize_to_tic=bool(normalize_to_tic))
    values = img[dataset.y_coords, dataset.x_coords]
    allowed = np.ones(len(values), dtype=bool)
    if prefilter_mask is not None:
        prefilter = np.asarray(prefilter_mask, dtype=bool)
        if prefilter.shape[0] != values.shape[0]:
            raise ValueError("prefilter_mask must match the number of MSI spectra.")
        allowed &= prefilter
    lower_selected = (values < float(threshold)) & allowed
    higher_selected = (values >= float(threshold)) & allowed
    if not np.any(lower_selected) and not np.any(higher_selected):
        raise ValueError("Threshold selected no MSI pixels.")

    transform = np.asarray(transform_xy if transform_xy is not None else np.eye(3, dtype=float), dtype=float)
    if transform.shape != (3, 3) or not np.all(np.isfinite(transform)):
        raise ValueError("transform_xy must be a finite 3x3 affine matrix.")

    def mask_to_geometry(selected: np.ndarray):
        polygons = []
        for x, y in zip(dataset.x_coords[selected].astype(float), dataset.y_coords[selected].astype(float)):
            corners = np.array(
                [
                    [x - 0.5, y - 0.5, 1.0],
                    [x + 0.5, y - 0.5, 1.0],
                    [x + 0.5, y + 0.5, 1.0],
                    [x - 0.5, y + 0.5, 1.0],
                ],
                dtype=float,
            )
            transformed = (transform @ corners.T).T[:, :2]
            polygons.append(Polygon(transformed))
        if not polygons:
            return None
        geometry = unary_union(polygons)
        return None if geometry.is_empty else geometry

    label_prefix = str(annotation_label).strip()
    lower_label = f"{label_prefix} Lower".strip() if label_prefix else "Lower"
    higher_label = f"{label_prefix} Higher".strip() if label_prefix else "Higher"
    key_base = sanitize_name(annotation_name) or sanitize_name(f"msi_threshold_{float(target_mz):.4f}_{float(threshold):g}")
    key = f"anno_{key_base}"
    sdata = sd.read_zarr(host_zarr_path)
    if key in sdata.shapes:
        suffix = 2
        while f"{key}_{suffix}" in sdata.shapes:
            suffix += 1
        key = f"{key}_{suffix}"

    rows = []
    geometries = []
    for label, threshold_side, selected in (
        (lower_label, "lower", lower_selected),
        (higher_label, "higher", higher_selected),
    ):
        geometry = mask_to_geometry(selected)
        if geometry is None:
            continue
        rows.append(
            {
                "_annotation_label": label,
                "source": "msi_threshold",
                "target_mz": float(target_mz),
                "ppm_tolerance": float(ppm_tolerance),
                "threshold": float(threshold),
                "threshold_side": threshold_side,
                "normalize_to_tic": bool(normalize_to_tic),
                "prefilter_shape_key": str(prefilter_shape_key),
                "prefilter_region_label": str(prefilter_region_label),
                "n_pixels": int(np.count_nonzero(selected)),
            }
        )
        geometries.append(geometry)
    if not rows:
        raise ValueError("Threshold produced no annotation geometries.")
    gdf = gpd.GeoDataFrame(rows, geometry=geometries)
    shape_element = ShapesModel.parse(gdf)
    set_transformation(shape_element, Identity(), to_coordinate_system=registered_cs)
    root = zarr.open_group(str(host_zarr_path), mode="a", use_consolidated=False)
    shapes_root = root.require_group("shapes")
    if key in shapes_root:
        del shapes_root[key]
    write_shapes(shape_element, shapes_root.require_group(key))
    zarr.consolidate_metadata(str(host_zarr_path))
    return key


def _reference_channel_image(
    sdata,
    reference_key: str,
    channel_index: int = 0,
) -> np.ndarray:
    if reference_key not in sdata.images:
        raise KeyError(f"Reference image {reference_key!r} was not found.")
    image = sdata.images[reference_key]
    raw_arr = np.asarray(image)
    image_attrs = getattr(image, "attrs", {})
    image_dims = tuple(getattr(image, "dims", ()))
    source_channels = int(image_attrs.get("source_channels", 0)) if isinstance(image_attrs, Mapping) else 0
    channel_index = int(channel_index)

    if raw_arr.ndim == 2:
        if channel_index != 0:
            raise ValueError(f"Reference image {reference_key!r} has only one channel.")
        return np.asarray(raw_arr, dtype=float)

    if raw_arr.ndim != 3:
        raise ValueError(f"Reference image {reference_key!r} has unsupported shape {raw_arr.shape}.")

    if source_channels > 4:
        if image_dims == ("c", "y", "x") and raw_arr.shape[0] == source_channels:
            if channel_index < 0 or channel_index >= raw_arr.shape[0]:
                raise ValueError(f"Channel {channel_index} is outside reference image {reference_key!r}.")
            return np.asarray(raw_arr[channel_index], dtype=float)
        if image_dims == ("y", "x", "c") and raw_arr.shape[-1] == source_channels:
            if channel_index < 0 or channel_index >= raw_arr.shape[-1]:
                raise ValueError(f"Channel {channel_index} is outside reference image {reference_key!r}.")
            return np.asarray(raw_arr[..., channel_index], dtype=float)

    if image_dims == ("c", "y", "x") or (raw_arr.shape[0] > 4 and raw_arr.shape[-1] <= 4):
        if channel_index < 0 or channel_index >= raw_arr.shape[0]:
            raise ValueError(f"Channel {channel_index} is outside reference image {reference_key!r}.")
        return np.asarray(raw_arr[channel_index], dtype=float)

    if image_dims == ("y", "x", "c") or raw_arr.shape[-1] <= 4:
        if channel_index == 0 and raw_arr.shape[-1] in (3, 4):
            rgb = np.asarray(raw_arr[..., :3], dtype=float)
            return np.dot(rgb, np.array([0.2126, 0.7152, 0.0722], dtype=float))
        if channel_index < 0 or channel_index >= raw_arr.shape[-1]:
            raise ValueError(f"Channel {channel_index} is outside reference image {reference_key!r}.")
        return np.asarray(raw_arr[..., channel_index], dtype=float)

    if channel_index < 0 or channel_index >= raw_arr.shape[-1]:
        raise ValueError(f"Channel {channel_index} is outside reference image {reference_key!r}.")
    return np.asarray(raw_arr[..., channel_index], dtype=float)


def _reference_channel_count_and_names(sdata, reference_key: str) -> tuple[int, list[str]]:
    if reference_key not in sdata.images:
        raise KeyError(f"Reference image {reference_key!r} was not found.")
    image = sdata.images[reference_key]
    raw_arr = np.asarray(image)
    image_attrs = getattr(image, "attrs", {})
    image_dims = tuple(getattr(image, "dims", ()))
    raw_names = image_attrs.get("channel_names", []) if isinstance(image_attrs, Mapping) else []

    if raw_arr.ndim == 2:
        count = 1
    elif raw_arr.ndim == 3:
        source_channels = int(image_attrs.get("source_channels", 0)) if isinstance(image_attrs, Mapping) else 0
        if source_channels > 4:
            if image_dims == ("c", "y", "x") and raw_arr.shape[0] == source_channels:
                count = int(raw_arr.shape[0])
            elif image_dims == ("y", "x", "c") and raw_arr.shape[-1] == source_channels:
                count = int(raw_arr.shape[-1])
            else:
                count = int(source_channels)
        elif image_dims == ("c", "y", "x") or (raw_arr.shape[0] > 4 and raw_arr.shape[-1] <= 4):
            count = int(raw_arr.shape[0])
        elif image_dims == ("y", "x", "c") or raw_arr.shape[-1] <= 4:
            count = 1 if raw_arr.shape[-1] in (3, 4) else int(raw_arr.shape[-1])
        else:
            count = int(raw_arr.shape[-1])
    else:
        raise ValueError(f"Reference image {reference_key!r} has unsupported shape {raw_arr.shape}.")

    names = []
    for idx in range(count):
        if idx < len(raw_names) and str(raw_names[idx]).strip():
            names.append(str(raw_names[idx]).strip())
        else:
            names.append(f"{reference_key} ch {idx + 1}")
    return count, names


def _sample_reference_values_at_msi_pixels(
    reference_img: np.ndarray,
    dataset: CoregistrationDataset,
    transform_xy: np.ndarray,
) -> np.ndarray:
    transform = np.asarray(transform_xy, dtype=float)
    if transform.shape != (3, 3) or not np.all(np.isfinite(transform)):
        raise ValueError("transform_xy must be a finite 3x3 affine matrix.")
    if reference_img.ndim != 2:
        raise ValueError("reference_img must be a 2D intensity image.")

    img = np.asarray(reference_img, dtype=float)
    values = np.full(dataset.x_coords.shape[0], np.nan, dtype=float)
    height, width = img.shape

    centers = np.column_stack(
        [
            dataset.x_coords.astype(float),
            dataset.y_coords.astype(float),
            np.ones_like(dataset.x_coords, dtype=float),
        ]
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ref_centers = (transform @ centers.T).T[:, :2]
    finite_centers = np.all(np.isfinite(ref_centers), axis=1)
    nearest_x = np.zeros(ref_centers.shape[0], dtype=int)
    nearest_y = np.zeros(ref_centers.shape[0], dtype=int)
    nearest_x[finite_centers] = np.rint(ref_centers[finite_centers, 0]).astype(int)
    nearest_y[finite_centers] = np.rint(ref_centers[finite_centers, 1]).astype(int)

    for idx, (x, y) in enumerate(zip(dataset.x_coords.astype(float), dataset.y_coords.astype(float))):
        corners = np.array(
            [
                [x - 0.5, y - 0.5, 1.0],
                [x + 0.5, y - 0.5, 1.0],
                [x + 0.5, y + 0.5, 1.0],
                [x - 0.5, y + 0.5, 1.0],
            ],
            dtype=float,
        )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            poly_xy = (transform @ corners.T).T[:, :2]
        if not np.all(np.isfinite(poly_xy)):
            continue
        min_x = max(0, int(np.floor(np.min(poly_xy[:, 0]))))
        max_x = min(width - 1, int(np.ceil(np.max(poly_xy[:, 0]))))
        min_y = max(0, int(np.floor(np.min(poly_xy[:, 1]))))
        max_y = min(height - 1, int(np.ceil(np.max(poly_xy[:, 1]))))
        if min_x <= max_x and min_y <= max_y:
            yy, xx = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
            sample_points = np.column_stack([xx.ravel().astype(float), yy.ravel().astype(float)])
            inside = MplPath(poly_xy).contains_points(sample_points, radius=1e-9)
            if np.any(inside):
                local_values = img[yy.ravel()[inside], xx.ravel()[inside]]
                finite = local_values[np.isfinite(local_values)]
                if finite.size:
                    values[idx] = float(np.mean(finite))
                    continue

        ref_x = nearest_x[idx]
        ref_y = nearest_y[idx]
        if finite_centers[idx] and 0 <= ref_x < width and 0 <= ref_y < height and np.isfinite(img[ref_y, ref_x]):
            values[idx] = float(img[ref_y, ref_x])

    return values


def _normalize_annotation_inclusion_mode(inclusion_mode: str) -> str:
    mode = str(inclusion_mode).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "centre": "center",
        "center_point": "center",
        "centre_point": "center",
        "pixel_center": "center",
        "pixel_centre": "center",
        "any_overlap": "intersects",
        "overlap": "intersects",
        "intersection": "intersects",
        "intersect": "intersects",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"center", "intersects"}:
        raise ValueError("inclusion_mode must be either 'center' or 'intersects'.")
    return mode


def _annotation_mask_from_transformed_geometries(
    dataset: CoregistrationDataset,
    rois: gpd.GeoDataFrame,
    transform_xy: np.ndarray,
    *,
    inclusion_mode: str = "center",
    min_hole_area: float = 0.0,
) -> np.ndarray:
    mode = _normalize_annotation_inclusion_mode(inclusion_mode)
    transform = np.asarray(transform_xy, dtype=float)
    if transform.shape != (3, 3) or not np.all(np.isfinite(transform)):
        raise ValueError("transform_xy must be a finite 3x3 affine matrix.")
    if len(rois) == 0:
        return np.zeros(len(dataset.x_coords), dtype=bool)

    xy1 = np.column_stack(
        [dataset.x_coords.astype(float), dataset.y_coords.astype(float), np.ones_like(dataset.x_coords, dtype=float)]
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        xy_t = (transform @ xy1.T).T[:, :2]
    finite = np.all(np.isfinite(xy_t), axis=1)
    if not np.any(finite):
        return np.zeros(len(dataset.x_coords), dtype=bool)

    min_hole_area = float(min_hole_area)
    if min_hole_area < 0 or not np.isfinite(min_hole_area):
        raise ValueError("min_hole_area must be a non-negative finite value.")
    geometries = [
        _fill_small_polygon_holes(geom, min_hole_area) if min_hole_area > 0 else geom
        for geom in rois.geometry
        if geom is not None and not geom.is_empty
    ]
    geometry = unary_union(geometries)
    if geometry is None or geometry.is_empty:
        return np.zeros(len(dataset.x_coords), dtype=bool)

    selected_finite = np.zeros(np.count_nonzero(finite), dtype=bool)
    if mode == "center":
        points = np.array([Point(px, py) for px, py in xy_t[finite]], dtype=object)
        selected_finite = np.fromiter((geometry.covers(point) for point in points), dtype=bool, count=len(points))
    else:
        finite_indices = np.flatnonzero(finite)
        for out_idx, pixel_idx in enumerate(finite_indices):
            x = float(dataset.x_coords[pixel_idx])
            y = float(dataset.y_coords[pixel_idx])
            corners = np.array(
                [
                    [x - 0.5, y - 0.5, 1.0],
                    [x + 0.5, y - 0.5, 1.0],
                    [x + 0.5, y + 0.5, 1.0],
                    [x - 0.5, y + 0.5, 1.0],
                ],
                dtype=float,
            )
            poly_xy = (transform @ corners.T).T[:, :2]
            pixel_poly = Polygon(poly_xy)
            selected_finite[out_idx] = pixel_poly.is_valid and geometry.intersects(pixel_poly)

    selected = np.zeros(len(dataset.x_coords), dtype=bool)
    selected[finite] = selected_finite
    return selected


def _fill_small_polygon_holes(geom, min_hole_area: float):
    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        interiors = [
            ring for ring in geom.interiors
            if Polygon(ring).area >= min_hole_area
        ]
        return Polygon(geom.exterior, interiors)
    if geom.geom_type == "MultiPolygon":
        return MultiPolygon([_fill_small_polygon_holes(poly, min_hole_area) for poly in geom.geoms])
    return geom


def sample_reference_channel_values_at_msi_pixels(
    zarr_path: str | Path | CoregistrationDataset,
    *,
    reference_key: str = "hne",
    channel_index: int = 0,
    msi_dataset: str | None = None,
    table_key: str | None = None,
    tic_key: str | None = None,
    transform_xy: np.ndarray | None = None,
    registered_cs: str = "registered",
) -> np.ndarray:
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
    if transform_xy is None:
        transform_xy, _found = dataset.load_saved_registration_if_available()
    reference_img = _reference_channel_image(dataset.sdata, reference_key, int(channel_index))
    return _sample_reference_values_at_msi_pixels(reference_img, dataset, np.asarray(transform_xy, dtype=float))


def get_coregistered_msi_mask_image(*args, **kwargs):
    from .coreg_figures import get_coregistered_msi_mask_image as _impl

    return _impl(*args, **kwargs)


def get_coregistered_reference_image(*args, **kwargs):
    from .coreg_figures import get_coregistered_reference_image as _impl

    return _impl(*args, **kwargs)


def get_coregistered_ion_image(*args, **kwargs):
    from .coreg_figures import get_coregistered_ion_image as _impl

    return _impl(*args, **kwargs)


def get_coregistered_image_layers(*args, **kwargs):
    from .coreg_figures import get_coregistered_image_layers as _impl

    return _impl(*args, **kwargs)



def create_annotation_region_mask(
    zarr_path: str | Path | CoregistrationDataset,
    annotation_key: str,
    *,
    region_label: str = "(all regions)",
    msi_dataset: str | None = None,
    table_key: str | None = None,
    tic_key: str | None = None,
    transform_xy: np.ndarray | None = None,
    inclusion_mode: str = "center",
    min_hole_area: float = 0.0,
    registered_cs: str = "registered",
) -> np.ndarray:
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
    if annotation_key not in dataset.sdata.shapes:
        raise KeyError(f"Annotation shapes {annotation_key!r} were not found.")

    try:
        rois = dataset.sdata.transform_element_to_coordinate_system(annotation_key, registered_cs)
    except Exception:
        rois = dataset.sdata.shapes[annotation_key]

    if region_label != "(all regions)" and "_annotation_label" in rois.columns:
        wanted = str(region_label).strip()
        rois = rois[rois["_annotation_label"].astype(str).str.strip() == wanted]
    if len(rois) == 0:
        return np.zeros(len(dataset.x_coords), dtype=bool)

    if transform_xy is None:
        transform_xy, _found = dataset.load_saved_registration_if_available()
    return _annotation_mask_from_transformed_geometries(
        dataset,
        rois,
        np.asarray(transform_xy, dtype=float),
        inclusion_mode=inclusion_mode,
        min_hole_area=min_hole_area,
    )


def summarize_annotation_region_spectra(
    zarr_path: str | Path | CoregistrationDataset,
    annotation_key: str,
    *,
    region_label: str = "(all regions)",
    msi_dataset: str | None = None,
    table_key: str | None = None,
    tic_key: str | None = None,
    transform_xy: np.ndarray | None = None,
    inclusion_mode: str = "center",
    min_hole_area: float = 0.0,
    normalize_to_tic: bool = True,
    registered_cs: str = "registered",
) -> dict[str, np.ndarray | int]:
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
    selected = create_annotation_region_mask(
        dataset,
        annotation_key,
        region_label=region_label,
        transform_xy=transform_xy,
        inclusion_mode=inclusion_mode,
        min_hole_area=min_hole_area,
        registered_cs=registered_cs,
    )
    return dataset.summarize_region_spectra(selected, normalize_to_tic=bool(normalize_to_tic))


def summarize_msi_pixel_mask_spectra(
    zarr_path: str | Path | CoregistrationDataset,
    pixel_mask: np.ndarray,
    *,
    msi_dataset: str | None = None,
    table_key: str | None = None,
    tic_key: str | None = None,
    normalize_to_tic: bool = True,
    registered_cs: str = "registered",
) -> dict[str, np.ndarray | int]:
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
    return dataset.summarize_region_spectra(pixel_mask, normalize_to_tic=bool(normalize_to_tic))


@dataclass
class ReferenceThresholdMask:
    reference_key: str
    channel_index: int
    threshold: float
    mode: str
    values: np.ndarray
    allowed_mask: np.ndarray
    below_mask: np.ndarray
    above_mask: np.ndarray
    below_image: np.ndarray
    above_image: np.ndarray


@dataclass
class ReferenceThresholdSpectra:
    mask: ReferenceThresholdMask
    below_summary: dict[str, np.ndarray | int] | None
    above_summary: dict[str, np.ndarray | int] | None


def create_reference_threshold_mask(
    zarr_path: str | Path | CoregistrationDataset,
    *,
    reference_key: str = "hne",
    channel_index: int = 0,
    threshold: float | None = None,
    percentile: float | None = None,
    msi_dataset: str | None = None,
    table_key: str | None = None,
    tic_key: str | None = None,
    transform_xy: np.ndarray | None = None,
    prefilter_mask: np.ndarray | None = None,
    registered_cs: str = "registered",
) -> ReferenceThresholdMask:
    if (threshold is None) == (percentile is None):
        raise ValueError("Supply exactly one of `threshold` or `percentile`.")

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
    reference_img = _reference_channel_image(dataset.sdata, reference_key, int(channel_index))
    if transform_xy is None:
        transform_xy, _found = dataset.load_saved_registration_if_available()
    values = _sample_reference_values_at_msi_pixels(reference_img, dataset, np.asarray(transform_xy, dtype=float))

    allowed = np.isfinite(values)
    if prefilter_mask is not None:
        prefilter = np.asarray(prefilter_mask, dtype=bool).ravel()
        if prefilter.shape[0] != values.shape[0]:
            raise ValueError("prefilter_mask must match the number of MSI spectra.")
        allowed &= prefilter
    if not np.any(allowed):
        raise ValueError("No MSI pixels have finite fluorescence values after filtering.")

    if percentile is not None:
        pct = float(percentile)
        if not 0 <= pct <= 100:
            raise ValueError("percentile must be between 0 and 100.")
        threshold_value = float(np.percentile(values[allowed], pct))
        mode = "percentile"
    else:
        threshold_value = float(threshold)
        if not np.isfinite(threshold_value):
            raise ValueError("threshold must be finite.")
        mode = "absolute"

    below = (values < threshold_value) & allowed
    above = (values >= threshold_value) & allowed
    if not np.any(below) and not np.any(above):
        raise ValueError("Threshold selected no MSI pixels.")

    below_image = np.zeros((dataset.ny, dataset.nx), dtype=np.uint8)
    above_image = np.zeros((dataset.ny, dataset.nx), dtype=np.uint8)
    below_image[dataset.y_coords[below], dataset.x_coords[below]] = 1
    above_image[dataset.y_coords[above], dataset.x_coords[above]] = 1

    return ReferenceThresholdMask(
        reference_key=str(reference_key),
        channel_index=int(channel_index),
        threshold=threshold_value,
        mode=mode,
        values=values,
        allowed_mask=allowed,
        below_mask=below,
        above_mask=above,
        below_image=below_image,
        above_image=above_image,
    )


def summarize_reference_threshold_spectra(
    zarr_path: str | Path | CoregistrationDataset,
    *,
    reference_key: str = "hne",
    channel_index: int = 0,
    threshold: float | None = None,
    percentile: float | None = None,
    msi_dataset: str | None = None,
    table_key: str | None = None,
    tic_key: str | None = None,
    transform_xy: np.ndarray | None = None,
    prefilter_mask: np.ndarray | None = None,
    normalize_to_tic: bool = True,
    registered_cs: str = "registered",
) -> ReferenceThresholdSpectra:
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
    mask = create_reference_threshold_mask(
        dataset,
        reference_key=reference_key,
        channel_index=channel_index,
        threshold=threshold,
        percentile=percentile,
        transform_xy=transform_xy,
        prefilter_mask=prefilter_mask,
    )

    below_summary = (
        dataset.summarize_region_spectra(mask.below_mask, normalize_to_tic=bool(normalize_to_tic))
        if np.any(mask.below_mask)
        else None
    )
    above_summary = (
        dataset.summarize_region_spectra(mask.above_mask, normalize_to_tic=bool(normalize_to_tic))
        if np.any(mask.above_mask)
        else None
    )
    return ReferenceThresholdSpectra(
        mask=mask,
        below_summary=below_summary,
        above_summary=above_summary,
    )


def _sample_msi_values_at_msi_pixels(
    source_img: np.ndarray,
    source_dataset: CoregistrationDataset,
    source_transform_xy: np.ndarray,
    target_dataset: CoregistrationDataset,
    target_transform_xy: np.ndarray,
) -> np.ndarray:
    source_transform = np.asarray(source_transform_xy, dtype=float)
    target_transform = np.asarray(target_transform_xy, dtype=float)
    if source_transform.shape != (3, 3) or not np.all(np.isfinite(source_transform)):
        raise ValueError("source_transform_xy must be a finite 3x3 affine matrix.")
    if target_transform.shape != (3, 3) or not np.all(np.isfinite(target_transform)):
        raise ValueError("target_transform_xy must be a finite 3x3 affine matrix.")

    source = np.asarray(source_img, dtype=float)
    if source.ndim != 2:
        raise ValueError("source_img must be a 2D MSI ion image.")

    values = source[source_dataset.y_coords, source_dataset.x_coords]
    finite_source = np.isfinite(values)
    if not np.any(finite_source):
        return np.full(target_dataset.x_coords.shape[0], np.nan, dtype=float)

    source_xy1 = np.column_stack(
        [
            source_dataset.x_coords.astype(float),
            source_dataset.y_coords.astype(float),
            np.ones_like(source_dataset.x_coords, dtype=float),
        ]
    )
    target_lookup = {
        (int(x), int(y)): idx
        for idx, (x, y) in enumerate(zip(target_dataset.x_coords.astype(int), target_dataset.y_coords.astype(int)))
    }
    target_inverse = np.linalg.inv(target_transform)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        source_registered = (source_transform @ source_xy1.T).T
        target_xy = (target_inverse @ source_registered.T).T[:, :2]

    target_min_x = float(np.min(target_dataset.x_coords)) - 0.5
    target_max_x = float(np.max(target_dataset.x_coords)) + 0.5
    target_min_y = float(np.min(target_dataset.y_coords)) - 0.5
    target_max_y = float(np.max(target_dataset.y_coords)) + 0.5
    candidate = (
        finite_source
        & np.all(np.isfinite(target_xy), axis=1)
        & (target_xy[:, 0] >= target_min_x)
        & (target_xy[:, 0] <= target_max_x)
        & (target_xy[:, 1] >= target_min_y)
        & (target_xy[:, 1] <= target_max_y)
    )
    nearest_x = np.zeros(target_xy.shape[0], dtype=int)
    nearest_y = np.zeros(target_xy.shape[0], dtype=int)
    nearest_x[candidate] = np.rint(target_xy[candidate, 0]).astype(int)
    nearest_y[candidate] = np.rint(target_xy[candidate, 1]).astype(int)
    inside_target_pixel = np.zeros(target_xy.shape[0], dtype=bool)
    inside_target_pixel[candidate] = (
        (np.abs(target_xy[candidate, 0] - nearest_x[candidate].astype(float)) <= 0.5 + 1e-9)
        & (np.abs(target_xy[candidate, 1] - nearest_y[candidate].astype(float)) <= 0.5 + 1e-9)
    )

    sums = np.zeros(target_dataset.x_coords.shape[0], dtype=float)
    counts = np.zeros(target_dataset.x_coords.shape[0], dtype=int)
    for src_idx in np.flatnonzero(inside_target_pixel):
        target_idx = target_lookup.get((int(nearest_x[src_idx]), int(nearest_y[src_idx])))
        if target_idx is None:
            continue
        sums[target_idx] += float(values[src_idx])
        counts[target_idx] += 1

    out = np.full(target_dataset.x_coords.shape[0], np.nan, dtype=float)
    has_values = counts > 0
    out[has_values] = sums[has_values] / counts[has_values]
    return out


def create_pooled_msi_threshold_annotation(
    zarr_path: str | Path,
    *,
    source_table_key: str | None = None,
    source_tic_key: str | None = None,
    target_table_key: str | None = None,
    target_tic_key: str | None = None,
    target_mz: float,
    ppm_tolerance: float = 5.0,
    threshold: float,
    normalize_to_tic: bool = True,
    source_transform_xy: np.ndarray | None = None,
    target_transform_xy: np.ndarray | None = None,
    prefilter_mask: np.ndarray | None = None,
    prefilter_shape_key: str = "",
    prefilter_region_label: str = "",
    annotation_name: str = "",
    annotation_label: str = "",
    registered_cs: str = "registered",
) -> str:
    host_zarr_path = Path(zarr_path).expanduser()
    source_dataset = CoregistrationDataset(host_zarr_path, registered_cs=registered_cs, table_key=source_table_key, tic_key=source_tic_key)
    target_dataset = CoregistrationDataset(host_zarr_path, registered_cs=registered_cs, table_key=target_table_key, tic_key=target_tic_key)
    indices = source_dataset.find_feature_indices_from_mz(float(target_mz), float(ppm_tolerance))
    if indices.size == 0:
        idx, _ = source_dataset.find_feature_idx_from_mz(float(target_mz), float("inf"))
        if idx is None:
            raise ValueError(f"No m/z feature found near {target_mz:g} in source MSI dataset.")
        indices = np.array([idx], dtype=int)

    source_img = source_dataset.reconstruct_ion_image(indices, normalize_to_tic=bool(normalize_to_tic))
    source_transform = np.asarray(source_transform_xy if source_transform_xy is not None else np.eye(3, dtype=float), dtype=float)
    target_transform = np.asarray(target_transform_xy if target_transform_xy is not None else np.eye(3, dtype=float), dtype=float)
    values = _sample_msi_values_at_msi_pixels(
        source_img,
        source_dataset,
        source_transform,
        target_dataset,
        target_transform,
    )
    allowed = np.isfinite(values)
    if prefilter_mask is not None:
        prefilter = np.asarray(prefilter_mask, dtype=bool)
        if prefilter.shape[0] != values.shape[0]:
            raise ValueError("prefilter_mask must match the number of target MSI spectra.")
        allowed &= prefilter
    lower_selected = (values < float(threshold)) & allowed
    higher_selected = (values >= float(threshold)) & allowed
    if not np.any(lower_selected) and not np.any(higher_selected):
        raise ValueError("Threshold selected no target MSI pixels.")

    def mask_to_geometry(selected: np.ndarray):
        polygons = []
        for x, y in zip(target_dataset.x_coords[selected].astype(float), target_dataset.y_coords[selected].astype(float)):
            corners = np.array(
                [
                    [x - 0.5, y - 0.5, 1.0],
                    [x + 0.5, y - 0.5, 1.0],
                    [x + 0.5, y + 0.5, 1.0],
                    [x - 0.5, y + 0.5, 1.0],
                ],
                dtype=float,
            )
            transformed = (target_transform @ corners.T).T[:, :2]
            polygons.append(Polygon(transformed))
        if not polygons:
            return None
        geometry = unary_union(polygons)
        return None if geometry.is_empty else geometry

    label_prefix = str(annotation_label).strip()
    lower_label = f"{label_prefix} Lower".strip() if label_prefix else "Lower"
    higher_label = f"{label_prefix} Higher".strip() if label_prefix else "Higher"
    source_label = sanitize_name(source_dataset.display_name or source_dataset.table_key)
    target_label = sanitize_name(target_dataset.display_name or target_dataset.table_key)
    key_base = sanitize_name(annotation_name) or sanitize_name(f"pooled_msi_threshold_{source_label}_to_{target_label}_{float(target_mz):.4f}_{float(threshold):g}")
    key = f"anno_{key_base}"
    sdata = sd.read_zarr(host_zarr_path)
    if key in sdata.shapes:
        suffix = 2
        while f"{key}_{suffix}" in sdata.shapes:
            suffix += 1
        key = f"{key}_{suffix}"

    rows = []
    geometries = []
    for label, threshold_side, selected in (
        (lower_label, "lower", lower_selected),
        (higher_label, "higher", higher_selected),
    ):
        geometry = mask_to_geometry(selected)
        if geometry is None:
            continue
        rows.append(
            {
                "_annotation_label": label,
                "source": "pooled_msi_threshold",
                "source_table_key": str(source_dataset.table_key),
                "target_table_key": str(target_dataset.table_key),
                "target_mz": float(target_mz),
                "ppm_tolerance": float(ppm_tolerance),
                "threshold": float(threshold),
                "threshold_side": threshold_side,
                "normalize_to_tic": bool(normalize_to_tic),
                "pooling": "mean_source_pixel_centers_in_target_pixel_window",
                "prefilter_shape_key": str(prefilter_shape_key),
                "prefilter_region_label": str(prefilter_region_label),
                "n_pixels": int(np.count_nonzero(selected)),
            }
        )
        geometries.append(geometry)
    if not rows:
        raise ValueError("Threshold produced no annotation geometries.")
    gdf = gpd.GeoDataFrame(rows, geometry=geometries)
    shape_element = ShapesModel.parse(gdf)
    set_transformation(shape_element, Identity(), to_coordinate_system=registered_cs)
    root = zarr.open_group(str(host_zarr_path), mode="a", use_consolidated=False)
    shapes_root = root.require_group("shapes")
    if key in shapes_root:
        del shapes_root[key]
    write_shapes(shape_element, shapes_root.require_group(key))
    zarr.consolidate_metadata(str(host_zarr_path))
    return key


def create_reference_threshold_annotation(
    zarr_path: str | Path,
    *,
    table_key: str | None = None,
    tic_key: str | None = None,
    reference_key: str,
    channel_index: int = 0,
    channel_name: str = "",
    threshold: float,
    transform_xy: np.ndarray | None = None,
    prefilter_mask: np.ndarray | None = None,
    prefilter_shape_key: str = "",
    prefilter_region_label: str = "",
    annotation_name: str = "",
    annotation_label: str = "",
    registered_cs: str = "registered",
) -> str:
    host_zarr_path = Path(zarr_path).expanduser()
    dataset = CoregistrationDataset(host_zarr_path, registered_cs=registered_cs, table_key=table_key, tic_key=tic_key)
    sdata = sd.read_zarr(host_zarr_path)
    reference_img = _reference_channel_image(sdata, reference_key, int(channel_index))
    transform = np.asarray(transform_xy if transform_xy is not None else np.eye(3, dtype=float), dtype=float)
    values = _sample_reference_values_at_msi_pixels(reference_img, dataset, transform)
    allowed = np.isfinite(values)
    if prefilter_mask is not None:
        prefilter = np.asarray(prefilter_mask, dtype=bool)
        if prefilter.shape[0] != values.shape[0]:
            raise ValueError("prefilter_mask must match the number of MSI spectra.")
        allowed &= prefilter
    lower_selected = (values < float(threshold)) & allowed
    higher_selected = (values >= float(threshold)) & allowed
    if not np.any(lower_selected) and not np.any(higher_selected):
        raise ValueError("Threshold selected no MSI pixels.")

    def mask_to_geometry(selected: np.ndarray):
        polygons = []
        for x, y in zip(dataset.x_coords[selected].astype(float), dataset.y_coords[selected].astype(float)):
            corners = np.array(
                [
                    [x - 0.5, y - 0.5, 1.0],
                    [x + 0.5, y - 0.5, 1.0],
                    [x + 0.5, y + 0.5, 1.0],
                    [x - 0.5, y + 0.5, 1.0],
                ],
                dtype=float,
            )
            transformed = (transform @ corners.T).T[:, :2]
            polygons.append(Polygon(transformed))
        if not polygons:
            return None
        geometry = unary_union(polygons)
        return None if geometry.is_empty else geometry

    label_prefix = str(annotation_label).strip()
    lower_label = f"{label_prefix} Lower".strip() if label_prefix else "Lower"
    higher_label = f"{label_prefix} Higher".strip() if label_prefix else "Higher"
    channel_label = str(channel_name).strip() or f"{reference_key}_ch_{int(channel_index) + 1}"
    key_base = sanitize_name(annotation_name) or sanitize_name(f"if_threshold_{channel_label}_{float(threshold):g}")
    key = f"anno_{key_base}"
    if key in sdata.shapes:
        suffix = 2
        while f"{key}_{suffix}" in sdata.shapes:
            suffix += 1
        key = f"{key}_{suffix}"

    rows = []
    geometries = []
    for label, threshold_side, selected in (
        (lower_label, "lower", lower_selected),
        (higher_label, "higher", higher_selected),
    ):
        geometry = mask_to_geometry(selected)
        if geometry is None:
            continue
        rows.append(
            {
                "_annotation_label": label,
                "source": "reference_threshold",
                "reference_key": str(reference_key),
                "channel_index": int(channel_index),
                "channel_name": channel_label,
                "threshold": float(threshold),
                "threshold_side": threshold_side,
                "reference_pooling": "mean",
                "prefilter_shape_key": str(prefilter_shape_key),
                "prefilter_region_label": str(prefilter_region_label),
                "n_pixels": int(np.count_nonzero(selected)),
            }
        )
        geometries.append(geometry)
    if not rows:
        raise ValueError("Threshold produced no annotation geometries.")
    gdf = gpd.GeoDataFrame(rows, geometry=geometries)
    shape_element = ShapesModel.parse(gdf)
    set_transformation(shape_element, Identity(), to_coordinate_system=registered_cs)
    root = zarr.open_group(str(host_zarr_path), mode="a", use_consolidated=False)
    shapes_root = root.require_group("shapes")
    if key in shapes_root:
        del shapes_root[key]
    write_shapes(shape_element, shapes_root.require_group(key))
    zarr.consolidate_metadata(str(host_zarr_path))
    return key


def save_coregistration(
    zarr_path: str | Path,
    transform_xy: np.ndarray,
    *,
    table_key: str | None = None,
    tic_key: str | None = None,
    registered_cs: str = "registered",
) -> dict[str, Any]:

    dataset = CoregistrationDataset(zarr_path, registered_cs=registered_cs, table_key=table_key, tic_key=tic_key)
    transform_xy = np.asarray(transform_xy, dtype=float)
    if transform_xy.shape != (3, 3):
        raise ValueError("`transform_xy` must be a 3x3 affine matrix in xy order.")

    for key in dataset.reference_image_keys:
        set_transformation(
            dataset.sdata.images[key],
            Identity(),
            to_coordinate_system=registered_cs,
            write_to_sdata=dataset.sdata,
        )

    msi_transform = Affine(
        matrix=transform_xy,
        input_axes=("x", "y"),
        output_axes=("x", "y"),
    )
    set_transformation(
        dataset.sdata.images[dataset.tic_key],
        msi_transform,
        to_coordinate_system=registered_cs,
        write_to_sdata=dataset.sdata,
    )

    for key in dataset.pixel_shape_keys:
        set_transformation(
            dataset.sdata.shapes[key],
            msi_transform,
            to_coordinate_system=registered_cs,
            write_to_sdata=dataset.sdata,
        )

    params = {
        "dataset_label": dataset.dataset_label,
        "table_key": dataset.table_key,
        "tic_key": dataset.tic_key,
        "coordinate_system": registered_cs,
        "sx": float(transform_xy[0, 0]),
        "sy": float(transform_xy[1, 1]),
        "tx": float(transform_xy[0, 2]),
        "ty": float(transform_xy[1, 2]),
        "affine_xy_3x3": transform_xy.tolist(),
    }
    dataset.msi_table.uns["coregistration_params"] = params

    dataset.sdata.write_transformations(dataset.tic_key)
    for key in dataset.reference_image_keys:
        if key in dataset.sdata.images:
            dataset.sdata.write_transformations(key)
    for key in dataset.pixel_shape_keys:
        dataset.sdata.write_transformations(key)

    dataset.sdata.write_consolidated_metadata()
    return params


def prepare_coregistration_zarr(
    *,
    input_path: str | Path | None = None,
    zarr_path: str | Path | None = None,
    optical_image_path: str | Path | None = None,
    hne_image_path: str | Path | None = None,
    annotation_paths: Iterable[str | Path] | None = None,
    registered_cs: str = "registered",
    converter: Any | None = None,
) -> Path:
    if zarr_path is None:
        if input_path is None:
            raise ValueError("Provide either `zarr_path` or `input_path`.")
        zarr = convert_input_to_zarr(input_path, converter=converter)
    else:
        zarr = Path(zarr_path).expanduser()
        if not zarr.exists():
            if input_path is None:
                raise FileNotFoundError(f"Zarr path does not exist: {zarr}")
            zarr = convert_input_to_zarr(input_path, zarr, converter=converter)

    if optical_image_path is not None:
        add_reference_image(zarr, optical_image_path, key="optical", registered_cs=registered_cs)
    if hne_image_path is not None:
        add_reference_image(zarr, hne_image_path, key="hne", registered_cs=registered_cs)
    if annotation_paths:
        import_geojson_annotations(
            zarr,
            annotation_paths,
            target_image="hne" if hne_image_path is not None else "optical",
            registered_cs=registered_cs,
        )
    return zarr


def prepare_coregistration_batch(
    jobs: Iterable[Mapping[str, Any]],
    *,
    registered_cs: str = "registered",
    converter: Any | None = None,
) -> list[Path]:
    outputs = []
    for job in jobs:
        outputs.append(
            prepare_coregistration_zarr(
                input_path=job.get("input_path"),
                zarr_path=job.get("zarr_path"),
                optical_image_path=job.get("optical_image_path"),
                hne_image_path=job.get("hne_image_path"),
                annotation_paths=job.get("annotation_paths"),
                registered_cs=job.get("registered_cs", registered_cs),
                converter=converter,
            )
        )
    return outputs


def launch_coregistration_gui(*args, **kwargs):
    from .coreg_gui import launch_coregistration_gui as _impl

    return _impl(*args, **kwargs)
