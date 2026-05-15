from __future__ import annotations

import json
import sys
import tempfile
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
import napari
import tifffile
from magicgui import magicgui
from matplotlib import colormaps as mpl_colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from napari.utils.colormaps import Colormap
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QAction, QIcon
from qtpy.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from shapely.geometry import Point
from shapely import affinity
from spatialdata._io import write_image, write_shapes, write_table


import numpy as np
import zarr

_QT_APP = None


def _ensure_qapplication(QApplication):
    global _QT_APP
    app = QApplication.instance()
    if app is None:
        _QT_APP = QApplication(sys.argv)
        app = _QT_APP
    return app


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


def _make_reference_channel_colormap(rgb: tuple[float, float, float], name: str) -> Colormap:
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
            return np.eye(3, dtype=float), None

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
    source_path = Path(image_path).expanduser()
    qptiff_meta: dict[str, float | int | str] = {}
    if source_path.suffix.lower() in {".qptiff", ".ome.tiff", ".ome.tif"} and qptiff_level is not None:
        img, qptiff_meta = _read_qptiff_image(source_path, level=int(qptiff_level))
    else:
        img = iio.imread(source_path)
    element = _parse_image_to_spatial(img)

    px_um_x = 2.54
    px_um_y = 2.54
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
            if dy > 0:
                px_um_y = 25400.0 / dy
    except Exception:
        pass

    if qptiff_meta:
        px_um_x *= float(qptiff_meta["image_to_fullres_scale_x"])
        px_um_y *= float(qptiff_meta["image_to_fullres_scale_y"])

    element.attrs["pixel_size_x_um"] = float(px_um_x)
    element.attrs["pixel_size_y_um"] = float(px_um_y)
    element.attrs["pixel_size_source"] = "image_metadata_or_default_10000dpi"
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
    sdata = sd.read_zarr(host_zarr_path)
    deleted: list[str] = []
    for key in keys:
        if key not in sdata.shapes:
            continue
        sdata.delete_element_from_disk(key)
        deleted.append(key)
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


def _pick_input_or_convert(default_zarr_path: str | Path | None = None) -> Path:

    _ensure_qapplication(QApplication)

    if default_zarr_path is not None:
        zarr = Path(default_zarr_path).expanduser()
        if zarr.exists():
            return zarr

    chooser = QMessageBox()
    chooser.setWindowTitle("Choose Startup Input")
    chooser.setText("Select a SpatialData .zarr folder, or an .imzML/.npz file to convert.")
    btn_zarr = chooser.addButton("Open .zarr Folder", QMessageBox.AcceptRole)
    btn_file = chooser.addButton("Open .imzML/.npz File", QMessageBox.ActionRole)
    chooser.addButton(QMessageBox.Cancel)
    chooser.exec_()
    clicked = chooser.clickedButton()

    if clicked is btn_zarr:
        selected_dir = QFileDialog.getExistingDirectory(None, "Select SpatialData .zarr folder", "")
        if not selected_dir:
            raise SystemExit("No input selected.")
        src = Path(selected_dir).expanduser()
        if src.suffix.lower() != ".zarr":
            raise ValueError(f"Selected folder is not a .zarr store: {src}")
        return src

    if clicked is btn_file:
        selected, _ = QFileDialog.getOpenFileName(
            None,
            "Select .imzML or .npz input",
            "",
            "MSI input files (*.imzML *.npz);;imzML (*.imzML);;NPZ (*.npz);;All files (*)",
        )
        if not selected:
            raise SystemExit("No input selected.")
        src = Path(selected).expanduser()
        out, _ = QFileDialog.getSaveFileName(
            None,
            "Save converted zarr as",
            str(src.with_suffix(".zarr")),
            "SpatialData zarr (*.zarr);;All files (*)",
        )
        if not out:
            raise SystemExit("No output zarr selected.")
        return convert_input_to_zarr(src, out)

    raise SystemExit("No input selected.")


def launch_coregistration_gui(
    zarr_path: str | Path | None = None,
    *,
    input_path: str | Path | None = None,
    registered_cs: str = "registered",
):

    if zarr_path is None:
        if input_path is not None:
            zarr_path = prepare_coregistration_zarr(input_path=input_path, registered_cs=registered_cs)
        else:
            zarr_path = _pick_input_or_convert()

    host_zarr_path = Path(zarr_path).expanduser()
    dataset = CoregistrationDataset(host_zarr_path, registered_cs=registered_cs)

    _ensure_qapplication(QApplication)

    def make_transparent_zero_colormap(cmap_name: str):
        colors = np.asarray(mpl_colormaps[cmap_name](np.linspace(0, 1, 256)), dtype=float)
        colors[0, 3] = 0.0
        return Colormap(colors=colors, name=f"{cmap_name.lower()}_zero_transparent")

    def make_binary_overlay_colormap(rgb=(1.0, 0.1, 0.1), alpha=0.45):
        colors = np.array([[0.0, 0.0, 0.0, 0.0], [float(rgb[0]), float(rgb[1]), float(rgb[2]), float(alpha)]], dtype=float)
        return Colormap(colors=colors, name="binary_overlay")

    viewer = napari.Viewer()
    overlay_colormap_order = [
        "viridis",
        "magma",
        "inferno",
        "plasma",
        "cividis",
        "turbo",
        "Reds",
        "Blues",
        "Greens",
        "Purples",
        "Oranges",
        "Greys",
        "YlOrRd",
        "YlGnBu",
        "cubehelix",
    ]
    overlay_colormap_cache: dict[str, Any] = {}

    def get_overlay_colormap(cmap_name: str):
        if cmap_name not in overlay_colormap_cache:
            overlay_colormap_cache[cmap_name] = make_transparent_zero_colormap(cmap_name)
        return overlay_colormap_cache[cmap_name]

    roi_overlay_colormap = make_binary_overlay_colormap()

    def choose_dataset_label(base_value: str | Path, existing: set[str]) -> str:
        base = _sanitize_dataset_label(base_value)
        candidate = base
        suffix = 2
        while candidate in existing:
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate

    reference_layers = {}
    datasets: dict[str, dict[str, Any]] = {}
    active_dataset_label: str | None = None
    dataset_choice_to_key: dict[str, str] = {}
    startup_camera_state: dict[str, Any] | None = None

    def dataset_choice_text(state: dict[str, Any]) -> str:
        label = str(state["label"]).strip() or str(state["id"])
        return label if label == state["id"] else f"{label} [{state['id']}]"

    def current_dataset_choices() -> list[str]:
        return [dataset_choice_text(state) for state in datasets.values()]

    def clear_layout(layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                clear_layout(child_layout)

    def build_dataset_state(coreg_dataset: CoregistrationDataset, label: str) -> dict[str, Any]:
        initial_idx = int(np.abs(coreg_dataset.mz_values - 268.1040).argmin())
        initial_img = coreg_dataset.reconstruct_ion_image(initial_idx, normalize_to_tic=True)

        saved_xy_matrix, found_registration = coreg_dataset.load_saved_registration_if_available()
        if found_registration and np.allclose(saved_xy_matrix, np.eye(3, dtype=float), atol=1e-9):
            found_registration = False
        if not found_registration:
            scale_guess, _ = coreg_dataset.estimate_initial_scale_from_pixel_sizes()
            saved_xy_matrix = scale_guess

        overlay_name = "viridis"
        ion_layer = viewer.add_image(
            prepare_ion_for_display(initial_img),
            name=f"{label} m/z {coreg_dataset.mz_values[initial_idx]:.4f}",
            opacity=0.6,
            colormap=get_overlay_colormap(overlay_name),
            contrast_limits=auto_contrast_limits(initial_img),
            visible=False,
        )
        msi_landmarks = viewer.add_points(name=f"{label} MSI landmarks", ndim=2, face_color="#ff7f0e", size=8, visible=False)
        if hasattr(msi_landmarks, "border_color"):
            msi_landmarks.border_color = "white"
        if hasattr(msi_landmarks, "editable"):
            msi_landmarks.editable = True
        if hasattr(msi_landmarks, "mode"):
            msi_landmarks.mode = "add"

        state = {
            "id": coreg_dataset.table_key,
            "label": label,
            "dataset": coreg_dataset,
            "current_feature_idx": initial_idx,
            "current_target_mz": float(coreg_dataset.mz_values[initial_idx]),
            "current_ppm_tolerance": 5.0,
            "current_feature_indices": np.array([initial_idx], dtype=int),
            "current_normalize_to_tic": True,
            "current_colormap_name": overlay_name,
            "current_opacity": 0.6,
            "current_contrast_low_pct": 1.0,
            "current_contrast_high_pct": 99.5,
            "current_transform_xy": np.asarray(saved_xy_matrix, dtype=float).copy(),
            "ion_layer": ion_layer,
            "roi_mask_layer": None,
            "selected_annotation_mask_layer": None,
            "msi_landmarks": msi_landmarks,
        }
        return state

    def apply_percentile_contrast_to_active_layer(img: np.ndarray):
        state = get_active_state()
        state["ion_layer"].contrast_limits = auto_contrast_limits(
            img,
            low_pct=float(state["current_contrast_low_pct"]),
            high_pct=float(state["current_contrast_high_pct"]),
        )

    def apply_transform_to_state(state: dict[str, Any]):
        aff_yx = xy_to_yx_matrix(state["current_transform_xy"])
        for layer_key in ("ion_layer",):
            layer = state[layer_key]
            layer.affine = aff_yx
            layer.scale = (1.0, 1.0)
            layer.translate = (0.0, 0.0)
        for layer_key in ("roi_mask_layer", "selected_annotation_mask_layer"):
            overlay_layer = state.get(layer_key)
            if overlay_layer is not None:
                overlay_layer.affine = aff_yx
                overlay_layer.scale = (1.0, 1.0)
                overlay_layer.translate = (0.0, 0.0)

    def ensure_roi_mask_layer(state: dict[str, Any]):
        if state.get("roi_mask_layer") is not None:
            return state["roi_mask_layer"]
        coreg_dataset = state["dataset"]
        roi_mask_layer = viewer.add_image(
            np.zeros((coreg_dataset.ny, coreg_dataset.nx), dtype=np.uint8),
            name=f"{state['label']} ROI selection mask",
            colormap=roi_overlay_colormap,
            contrast_limits=(0, 1),
            interpolation2d="nearest",
            blending="translucent",
            opacity=1.0,
            visible=False,
        )
        state["roi_mask_layer"] = roi_mask_layer
        apply_transform_to_state(state)
        return roi_mask_layer

    def ensure_selected_annotation_mask_layer(state: dict[str, Any]):
        if state.get("selected_annotation_mask_layer") is not None:
            return state["selected_annotation_mask_layer"]
        coreg_dataset = state["dataset"]
        selected_mask_layer = viewer.add_image(
            np.zeros((coreg_dataset.ny, coreg_dataset.nx), dtype=np.uint8),
            name=f"{state['label']} selected annotation mask",
            colormap=roi_overlay_colormap,
            contrast_limits=(0, 1),
            interpolation2d="nearest",
            blending="translucent",
            opacity=1.0,
            visible=False,
        )
        state["selected_annotation_mask_layer"] = selected_mask_layer
        apply_transform_to_state(state)
        return selected_mask_layer

    def _reference_layer_list(key: str) -> list[Any]:
        layers = reference_layers.get(key)
        if layers is None:
            return []
        return layers if isinstance(layers, list) else [layers]

    def _reference_channel_layers() -> list[Any]:
        layers: list[Any] = []
        for key in reference_layers:
            layers.extend(_reference_layer_list(key))
        return layers

    def _reference_layer_count() -> int:
        return sum(len(_reference_layer_list(key)) for key in reference_layers)

    def _reference_channel_choice_names() -> list[str]:
        names = [str(layer.name) for layer in _reference_channel_layers()]
        return names if names else ["(none)"]

    def _get_reference_layer_by_name(name: str):
        for layer in _reference_channel_layers():
            if str(layer.name) == str(name):
                return layer
        return None

    def _layer_metadata(layer) -> dict[str, Any]:
        metadata = getattr(layer, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            try:
                layer.metadata = metadata
            except Exception:
                pass
        return metadata

    def _reference_color_choices() -> list[str]:
        return list(REFERENCE_CHANNEL_COLOR_PRESETS.keys())

    def _set_reference_layer_color(layer, color_choice: str):
        metadata = _layer_metadata(layer)
        choice = str(color_choice).strip().lower() or "metadata"
        rgb = metadata.get("reference_default_rgb")
        if choice != "metadata" or not (isinstance(rgb, (tuple, list)) and len(rgb) == 3):
            rgb = REFERENCE_CHANNEL_COLOR_PRESETS.get(choice, REFERENCE_CHANNEL_COLOR_PRESETS["white"])
        metadata["reference_color_choice"] = choice
        try:
            layer.colormap = _make_reference_channel_colormap(
                tuple(float(v) for v in rgb),
                f"{sanitize_name(str(layer.name))}_{choice}_cmap",
            )
        except Exception:
            pass

    def _refresh_if_toolbox_widgets(preferred_layer_name: str | None = None):
        choices = _reference_channel_choice_names()
        try:
            editor_widget = if_channel_editor
        except NameError:
            editor_widget = None
        if editor_widget is not None:
            current_value = str(if_channel_editor.layer_name.value)
            if_channel_editor.layer_name.choices = choices
            target_value = preferred_layer_name or current_value
            if target_value not in choices:
                target_value = choices[0]
            if_channel_editor.layer_name.value = target_value
        try:
            visibility_widget = if_channel_visibility_widget
        except NameError:
            visibility_widget = None
        if visibility_widget is not None:
            current_value = str(if_channel_visibility_widget.layer_name.value)
            if_channel_visibility_widget.layer_name.choices = choices
            target_value = preferred_layer_name or current_value
            if target_value not in choices:
                target_value = choices[0]
            if_channel_visibility_widget.layer_name.value = target_value
        try:
            layout = if_layer_controls_layout
        except NameError:
            layout = None
        if layout is not None:
            rebuild_if_layer_controls()

    def enforce_reference_layers_at_bottom():
        ordered = [key for key in ("optical", "hne") if key in reference_layers]
        ordered.extend([key for key in reference_layers if key not in ordered])
        target_idx = 0
        for key in ordered:
            for layer in _reference_layer_list(key):
                try:
                    current_idx = viewer.layers.index(layer)
                except Exception:
                    continue
                if current_idx != target_idx:
                    viewer.layers.move(current_idx, target_idx)
                target_idx += 1

    def add_or_update_reference_layer(source_dataset: CoregistrationDataset, key: str, *, visible: bool = True):
        image = source_dataset.sdata.images[key]
        raw_arr = np.asarray(image)
        arr = _to_napari_image(raw_arr)
        image_attrs = getattr(image, "attrs", {})
        image_dims = tuple(getattr(image, "dims", ()))
        existing_layers = _reference_layer_list(key)
        source_channels = int(image_attrs.get("source_channels", 0)) if isinstance(image_attrs, Mapping) else 0
        inferred_channels = 0
        if raw_arr.ndim == 3:
            if source_channels > 4:
                inferred_channels = source_channels
            elif image_dims == ("c", "y", "x") and raw_arr.shape[0] > 4:
                inferred_channels = int(raw_arr.shape[0])
            elif image_dims == ("y", "x", "c") and raw_arr.shape[-1] > 4:
                inferred_channels = int(raw_arr.shape[-1])
            elif raw_arr.shape[0] > 4 and raw_arr.shape[-1] <= 4:
                inferred_channels = int(raw_arr.shape[0])

        if raw_arr.ndim == 3 and inferred_channels > 4:
            if image_dims == ("c", "y", "x") and raw_arr.shape[0] == inferred_channels:
                channel_data = [raw_arr[idx] for idx in range(inferred_channels)]
            elif image_dims == ("y", "x", "c") and raw_arr.shape[-1] == inferred_channels:
                channel_data = [raw_arr[..., idx] for idx in range(inferred_channels)]
            elif raw_arr.shape[0] == inferred_channels:
                channel_data = [raw_arr[idx] for idx in range(inferred_channels)]
            else:
                channel_data = [raw_arr[..., idx] for idx in range(raw_arr.shape[-1])]
            raw_names = image_attrs.get("channel_names", []) if isinstance(image_attrs, Mapping) else []
            raw_colors = image_attrs.get("channel_colors", []) if isinstance(image_attrs, Mapping) else []
            channel_names = [
                f"{key}: {raw_names[idx]}" if idx < len(raw_names) and str(raw_names[idx]).strip() else f"{key} ch {idx + 1}"
                for idx in range(len(channel_data))
            ]
            channel_colormaps = []
            for idx in range(len(channel_data)):
                use_rgb = None
                if idx < len(raw_colors) and isinstance(raw_colors[idx], (list, tuple)) and len(raw_colors[idx]) == 3:
                    candidate = tuple(float(v) for v in raw_colors[idx])
                    if any(v > 0.05 for v in candidate):
                        use_rgb = candidate
                if use_rgb is None:
                    use_rgb = _fallback_reference_channel_color(idx)
                channel_colormaps.append(
                    _make_reference_channel_colormap(use_rgb, f"{sanitize_name(channel_names[idx])}_cmap")
                )
            if len(existing_layers) != len(channel_data):
                for layer in existing_layers:
                    try:
                        viewer.layers.remove(layer)
                    except Exception:
                        pass
                existing_layers = []
            if existing_layers:
                for idx, layer in enumerate(existing_layers):
                    layer.data = channel_data[idx]
                    layer.name = channel_names[idx]
                    layer.visible = bool(visible and idx == 0)
                    layer.opacity = 1.0
                    layer.blending = "translucent"
                    try:
                        layer.contrast_limits = auto_contrast_limits(channel_data[idx], low_pct=1.0, high_pct=99.8)
                    except Exception:
                        pass
                    try:
                        layer.colormap = channel_colormaps[idx]
                    except Exception:
                        pass
            else:
                existing_layers = [
                    viewer.add_image(
                        data,
                        name=name,
                        visible=bool(visible and idx == 0),
                        blending="translucent",
                        colormap=channel_colormaps[idx],
                        contrast_limits=auto_contrast_limits(data, low_pct=1.0, high_pct=99.8),
                        opacity=1.0,
                    )
                    for idx, (data, name) in enumerate(zip(channel_data, channel_names))
                ]
            for idx, layer in enumerate(existing_layers):
                metadata = _layer_metadata(layer)
                default_rgb = tuple(float(v) for v in channel_colormaps[idx].colors[-1][:3])
                metadata["reference_key"] = key
                metadata["reference_channel_index"] = idx
                metadata["reference_default_name"] = channel_names[idx]
                metadata["reference_default_rgb"] = default_rgb
                metadata.setdefault("reference_color_choice", "metadata")
                _set_reference_layer_color(layer, str(metadata.get("reference_color_choice", "metadata")))
            reference_layers[key] = existing_layers
        else:
            layer = existing_layers[0] if existing_layers else None
            if layer is not None:
                layer.data = arr
                layer.visible = visible
            else:
                layer = viewer.add_image(arr, name=key, visible=visible)
            metadata = _layer_metadata(layer)
            metadata["reference_key"] = key
            metadata["reference_channel_index"] = 0
            metadata["reference_default_name"] = key
            metadata.setdefault("reference_color_choice", "metadata")
            if metadata.get("reference_default_rgb") is None:
                metadata["reference_default_rgb"] = REFERENCE_CHANNEL_COLOR_PRESETS["white"]
            _set_reference_layer_color(layer, str(metadata.get("reference_color_choice", "metadata")))
            reference_layers[key] = [layer]
        enforce_reference_layers_at_bottom()
        _refresh_if_toolbox_widgets(preferred_layer_name=str(_reference_layer_list(key)[0].name))

    def add_dataset_to_view(coreg_dataset: CoregistrationDataset, label: str):
        state = build_dataset_state(coreg_dataset, label)
        datasets[str(state["id"])] = state
        apply_transform_to_state(state)
        for idx, key in enumerate(coreg_dataset.reference_image_keys):
            add_or_update_reference_layer(coreg_dataset, key, visible=(idx == 0 and _reference_layer_count() == 0))
        return state

    for spec in _infer_msi_dataset_specs(dataset.sdata):
        embedded_dataset = CoregistrationDataset(
            host_zarr_path,
            registered_cs=registered_cs,
            table_key=spec["table_key"],
            tic_key=spec["tic_key"],
        )
        add_dataset_to_view(embedded_dataset, str(embedded_dataset.display_name))

    annotation_shape_layers = {}
    annotation_edge_width = 1.5
    annotation_edge_opacity = 0.9
    annotation_show_labels = False
    annotation_label_size = 10

    def geometry_to_napari_shapes(geom):
        out = []
        if geom is None or geom.is_empty:
            return out
        if geom.geom_type == "Polygon":
            xy = np.asarray(geom.exterior.coords, dtype=float)
            if xy.shape[0] >= 3:
                out.append((xy[:, [1, 0]], "polygon"))
            return out
        if geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                out.extend(geometry_to_napari_shapes(poly))
            return out
        if geom.geom_type == "LineString":
            xy = np.asarray(geom.coords, dtype=float)
            if xy.shape[0] >= 2:
                out.append((xy[:, [1, 0]], "path"))
            return out
        if geom.geom_type == "MultiLineString":
            for line in geom.geoms:
                out.extend(geometry_to_napari_shapes(line))
        return out

    def apply_annotation_visuals():
        for layer in annotation_shape_layers.values():
            try:
                layer.edge_width = float(annotation_edge_width)
                layer.face_color = [0, 0, 0, 0]
                layer.opacity = float(annotation_edge_opacity)
                if annotation_show_labels:
                    layer.text.visible = True
                    layer.text.size = int(annotation_label_size)
                else:
                    layer.text.visible = False
            except Exception:
                pass

    def current_annotation_shape_keys() -> list[str]:
        keys: list[str] = []
        for state in datasets.values():
            for key in state["dataset"].sdata.shapes.keys():
                if "pixels" in key.lower() or not key.startswith("anno_"):
                    continue
                if key not in keys:
                    keys.append(key)
        return keys

    def annotation_region_choices(coreg_dataset: CoregistrationDataset, roi_shape_key: str) -> list[str]:
        if roi_shape_key not in coreg_dataset.sdata.shapes:
            return ["(all regions)"]
        try:
            rois = coreg_dataset.sdata.transform_element_to_coordinate_system(roi_shape_key, registered_cs)
        except Exception:
            rois = coreg_dataset.sdata.shapes[roi_shape_key]
        choices = ["(all regions)"]
        if "_annotation_label" not in rois.columns:
            return choices
        labels: list[str] = []
        for value in rois["_annotation_label"]:
            text = str(value).strip()
            if not text or text.lower() == "none":
                continue
            if text not in labels:
                labels.append(text)
        choices.extend(labels)
        return choices

    def compute_annotation_region_mask(
        state: dict[str, Any],
        roi_shape_key: str,
        region_label: str = "(all regions)",
    ) -> np.ndarray:
        coreg_dataset = state["dataset"]
        if roi_shape_key not in coreg_dataset.sdata.shapes:
            return np.zeros(len(coreg_dataset.x_coords), dtype=bool)
        rois = coreg_dataset.sdata.transform_element_to_coordinate_system(roi_shape_key, registered_cs)
        xy1 = np.column_stack(
            [coreg_dataset.x_coords.astype(float), coreg_dataset.y_coords.astype(float), np.ones_like(coreg_dataset.x_coords, dtype=float)]
        )
        xy_t = (state["current_transform_xy"] @ xy1.T).T[:, :2]
        points = np.array([Point(px, py) for px, py in xy_t], dtype=object)
        inside = [
            np.fromiter((point.within(rois.geometry.iloc[idx]) for point in points), dtype=bool, count=len(points))
            for idx in range(len(rois))
        ]
        inside = np.vstack(inside) if inside else np.zeros((0, len(points)), dtype=bool)
        if region_label == "(all regions)" or "_annotation_label" not in rois.columns:
            return np.any(inside, axis=0) if inside.size else np.zeros(len(points), dtype=bool)
        matching_idxs = [
            idx for idx, value in enumerate(rois["_annotation_label"])
            if str(value).strip() == str(region_label).strip()
        ]
        return np.any(inside[matching_idxs], axis=0) if matching_idxs else np.zeros(len(points), dtype=bool)

    def compute_selected_annotation_mask(active_layer, *, include_same_label: bool = True) -> tuple[dict[str, Any] | None, str | None, np.ndarray | None, str]:
        if active_layer is None or active_layer not in annotation_shape_layers.values():
            return None, None, None, ""
        selected_data = sorted(int(idx) for idx in getattr(active_layer, "selected_data", set()))
        if not selected_data:
            return None, None, None, ""
        metadata = _layer_metadata(active_layer)
        dataset_id = str(metadata.get("annotation_dataset_id", ""))
        shape_key = str(metadata.get("annotation_shape_key", ""))
        row_lookup = list(metadata.get("annotation_source_row_indices", []))
        if not dataset_id or not shape_key or not row_lookup:
            return None, None, None, ""
        source_state = datasets.get(dataset_id)
        if source_state is None:
            return None, None, None, ""
        source_gdf = source_state["dataset"].sdata.shapes[shape_key].copy()
        row_indices = sorted({int(row_lookup[idx]) for idx in selected_data if 0 <= int(idx) < len(row_lookup)})
        if include_same_label and "_annotation_label" in source_gdf.columns:
            selected_labels = {
                str(source_gdf.iloc[row_idx]["_annotation_label"]).strip()
                for row_idx in row_indices
                if 0 <= row_idx < len(source_gdf)
            }
            selected_labels = {label for label in selected_labels if label}
            if selected_labels:
                row_indices = [
                    idx for idx, value in enumerate(source_gdf["_annotation_label"])
                    if str(value).strip() in selected_labels
                ]
        if not row_indices:
            return source_state, shape_key, np.zeros(len(source_state["dataset"].x_coords), dtype=bool), ""
        default_label = ""
        if "_annotation_label" in source_gdf.columns:
            labels = [str(source_gdf.iloc[idx]["_annotation_label"]).strip() for idx in row_indices if str(source_gdf.iloc[idx]["_annotation_label"]).strip()]
            default_label = labels[0] if labels else ""
        transformed_gdf = source_state["dataset"].sdata.transform_element_to_coordinate_system(shape_key, registered_cs)
        transformed_subset = transformed_gdf.iloc[row_indices].copy()
        coreg_dataset = source_state["dataset"]
        xy1 = np.column_stack(
            [coreg_dataset.x_coords.astype(float), coreg_dataset.y_coords.astype(float), np.ones_like(coreg_dataset.x_coords, dtype=float)]
        )
        xy_t = (source_state["current_transform_xy"] @ xy1.T).T[:, :2]
        points = np.array([Point(px, py) for px, py in xy_t], dtype=object)
        inside = [
            np.fromiter((point.within(transformed_subset.geometry.iloc[idx]) for point in points), dtype=bool, count=len(points))
            for idx in range(len(transformed_subset))
        ]
        selected_mask = np.any(np.vstack(inside), axis=0) if inside else np.zeros(len(points), dtype=bool)
        return source_state, shape_key, selected_mask, default_label

    def refresh_annotation_widget_choices():
        keys = current_annotation_shape_keys()
        choices = ["(none)"] + keys if keys else ["(none)"]
        if "remove_geojson_annotations" in locals():
            remove_geojson_annotations.annotation_key.choices = choices
            if remove_geojson_annotations.annotation_key.value not in choices:
                remove_geojson_annotations.annotation_key.value = choices[0]
        if "rescale_geojson_annotations_widget" in locals():
            rescale_geojson_annotations_widget.annotation_key.choices = choices
            if rescale_geojson_annotations_widget.annotation_key.value not in choices:
                rescale_geojson_annotations_widget.annotation_key.value = choices[0]

    def add_annotation_shape_layers(state: dict[str, Any], shape_keys: Iterable[str] | None = None):
        source_dataset = state["dataset"]
        keys = [key for key in (shape_keys or source_dataset.sdata.shapes.keys()) if "pixels" not in key.lower()]
        colors = ["#ffcc00", "#00d1ff", "#ff5f87", "#7bff57", "#c18bff"]
        for idx, key in enumerate(keys):
            try:
                gdf = source_dataset.sdata.transform_element_to_coordinate_system(key, registered_cs)
            except Exception:
                gdf = source_dataset.sdata.shapes[key]
            shape_data = []
            shape_types = []
            shape_labels = []
            shape_row_indices = []
            for row_idx, geom in enumerate(gdf.geometry):
                label_text = ""
                if "_annotation_label" in gdf.columns:
                    try:
                        label_text = str(gdf.iloc[row_idx]["_annotation_label"]).strip()
                    except Exception:
                        label_text = ""
                for arr_yx, stype in geometry_to_napari_shapes(geom):
                    shape_data.append(arr_yx)
                    shape_types.append(stype)
                    shape_labels.append(label_text)
                    shape_row_indices.append(int(row_idx))
            if not shape_data:
                continue
            layer_name = f"anno:{state['label']}:{key}"
            if layer_name in annotation_shape_layers:
                annotation_shape_layers[layer_name].data = shape_data
                try:
                    annotation_shape_layers[layer_name].properties = {"label": np.asarray(shape_labels, dtype=object)}
                    annotation_shape_layers[layer_name].text = {
                        "string": "{label}",
                        "size": int(annotation_label_size),
                        "color": "white",
                        "anchor": "center",
                        "visible": bool(annotation_show_labels),
                    }
                except Exception:
                    pass
                metadata = _layer_metadata(annotation_shape_layers[layer_name])
                metadata["annotation_dataset_id"] = str(state["id"])
                metadata["annotation_shape_key"] = key
                metadata["annotation_source_row_indices"] = list(shape_row_indices)
                annotation_shape_layers[layer_name].visible = True
                continue
            annotation_shape_layers[layer_name] = viewer.add_shapes(
                shape_data,
                shape_type=shape_types,
                name=layer_name,
                edge_color=colors[idx % len(colors)],
                face_color=[0, 0, 0, 0],
                edge_width=float(annotation_edge_width),
                opacity=float(annotation_edge_opacity),
                blending="translucent",
                visible=True,
            )
            try:
                annotation_shape_layers[layer_name].properties = {"label": np.asarray(shape_labels, dtype=object)}
                annotation_shape_layers[layer_name].text = {
                    "string": "{label}",
                    "size": int(annotation_label_size),
                    "color": "white",
                    "anchor": "center",
                    "visible": bool(annotation_show_labels),
                }
            except Exception:
                pass
            metadata = _layer_metadata(annotation_shape_layers[layer_name])
            metadata["annotation_dataset_id"] = str(state["id"])
            metadata["annotation_shape_key"] = key
            metadata["annotation_source_row_indices"] = list(shape_row_indices)
        apply_annotation_visuals()
        refresh_annotation_widget_choices()

    def remove_annotation_shape_layers(shape_keys: Iterable[str]):
        remove_keys = set(str(key) for key in shape_keys)
        for layer_name in list(annotation_shape_layers.keys()):
            key = layer_name.split(":", 2)[-1]
            if key not in remove_keys:
                continue
            layer = annotation_shape_layers.pop(layer_name, None)
            if layer is None:
                continue
            try:
                viewer.layers.remove(layer)
            except Exception:
                pass
        refresh_annotation_widget_choices()

    initial_state = next(iter(datasets.values()))
    active_dataset_label = str(initial_state["id"])
    initial_state["ion_layer"].visible = True

    def get_active_state() -> dict[str, Any]:
        assert active_dataset_label is not None
        return datasets[active_dataset_label]

    def update_ion_view(feature_idx: int):
        state = get_active_state()
        coreg_dataset = state["dataset"]
        state["current_feature_idx"] = int(np.clip(feature_idx, 0, len(coreg_dataset.mz_values) - 1))
        state["current_feature_indices"] = np.array([state["current_feature_idx"]], dtype=int)
        state["current_target_mz"] = float(coreg_dataset.mz_values[state["current_feature_idx"]])
        img = coreg_dataset.reconstruct_ion_image(
            state["current_feature_indices"],
            normalize_to_tic=state["current_normalize_to_tic"],
        )
        state["ion_layer"].data = prepare_ion_for_display(img)
        state["ion_layer"].name = f"{state['label']} m/z {coreg_dataset.mz_values[state['current_feature_idx']]:.4f}"
        apply_percentile_contrast_to_active_layer(img)
        if current_mz_line is not None:
            current_mz_line.set_xdata(
                [coreg_dataset.mz_values[state["current_feature_idx"]], coreg_dataset.mz_values[state["current_feature_idx"]]]
            )
            spectrum_canvas.draw_idle()

    def update_ion_view_for_mz(target_mz: float, ppm_tolerance: float):
        state = get_active_state()
        coreg_dataset = state["dataset"]
        indices = coreg_dataset.find_feature_indices_from_mz(float(target_mz), float(ppm_tolerance))
        if indices.size == 0:
            idx, _ = coreg_dataset.find_feature_idx_from_mz(float(target_mz), float("inf"))
            if idx is None:
                return
            indices = np.array([idx], dtype=int)
        nearest_local = int(np.argmin(np.abs(coreg_dataset.mz_values[indices] - float(target_mz))))
        state["current_feature_idx"] = int(indices[nearest_local])
        state["current_feature_indices"] = np.asarray(indices, dtype=int)
        state["current_target_mz"] = float(target_mz)
        state["current_ppm_tolerance"] = float(ppm_tolerance)
        img = coreg_dataset.reconstruct_ion_image(
            state["current_feature_indices"],
            normalize_to_tic=state["current_normalize_to_tic"],
        )
        state["ion_layer"].data = prepare_ion_for_display(img)
        if len(state["current_feature_indices"]) > 1:
            state["ion_layer"].name = f"{state['label']} m/z {float(target_mz):.4f} +/- {float(ppm_tolerance):.1f} ppm"
        else:
            state["ion_layer"].name = f"{state['label']} m/z {coreg_dataset.mz_values[state['current_feature_idx']]:.4f}"
        apply_percentile_contrast_to_active_layer(img)
        if current_mz_line is not None:
            current_mz_line.set_xdata([float(target_mz), float(target_mz)])
            spectrum_canvas.draw_idle()

    spectrum_widget = QWidget()
    spectrum_layout = QVBoxLayout(spectrum_widget)
    spectrum_layout.setContentsMargins(6, 6, 6, 6)
    spectrum_label = QLabel("Average spectrum (click to select m/z)")
    spectrum_layout.addWidget(spectrum_label)
    spectrum_widget.setMinimumHeight(360)
    spectrum_widget.setMinimumWidth(560)
    spectrum_fig = Figure(figsize=(7.5, 3.8), constrained_layout=True)
    spectrum_canvas = FigureCanvas(spectrum_fig)
    spectrum_toolbar = NavigationToolbar2QT(spectrum_canvas, spectrum_widget)
    pick_mz_action = QAction("Pick m/z", spectrum_toolbar)
    pick_mz_action.setCheckable(True)
    pick_mz_action.setChecked(True)
    spectrum_toolbar.addSeparator()
    spectrum_toolbar.addAction(pick_mz_action)
    for action in list(spectrum_toolbar.actions()):
        text = str(action.text()).lower()
        if any(token in text for token in ("back", "forward", "subplots", "customize")):
            spectrum_toolbar.removeAction(action)
    spectrum_ax = spectrum_fig.add_subplot(111)
    current_mz_line = None
    spectrum_layout.addWidget(spectrum_toolbar)
    spectrum_layout.addWidget(spectrum_canvas)

    def recolor_toolbar_icons():
        fg_hex = "#ffffff"
        active_hex = "#d7191c"
        for child in spectrum_toolbar.findChildren(QToolButton):
            is_active = bool(child.isChecked())
            child.setStyleSheet(f"color: {active_hex if is_active else fg_hex};")
            icon = child.icon()
            if icon.isNull():
                continue
            pixmap = icon.pixmap(18, 18)
            if pixmap.isNull():
                continue
            mask = pixmap.createMaskFromColor(Qt.GlobalColor.transparent)
            pixmap.fill(Qt.GlobalColor.red if is_active else Qt.GlobalColor.white)
            pixmap.setMask(mask)
            child.setIcon(QIcon(pixmap))

    def schedule_toolbar_recolor(*_args):
        QTimer.singleShot(0, recolor_toolbar_icons)
        QTimer.singleShot(0, clamp_spectrum_ylim)

    def clamp_spectrum_ylim(_axes=None):
        ymin, ymax = spectrum_ax.get_ylim()
        if ymin < 0.0:
            spectrum_ax.set_ylim(bottom=0.0, top=max(ymax, 0.0))
            spectrum_canvas.draw_idle()

    def apply_spectrum_theme():
        fg_hex = "#ffffff"
        spectrum_widget.setStyleSheet(f"background-color: transparent; color: {fg_hex};")
        spectrum_label.setStyleSheet(f"color: {fg_hex};")
        spectrum_canvas.setStyleSheet("background: transparent;")
        spectrum_fig.patch.set_alpha(0.0)
        spectrum_ax.set_facecolor((0.0, 0.0, 0.0, 0.0))
        spectrum_ax.xaxis.label.set_color(fg_hex)
        spectrum_ax.yaxis.label.set_color(fg_hex)
        spectrum_ax.title.set_color(fg_hex)
        spectrum_ax.tick_params(colors=fg_hex)
        for spine in spectrum_ax.spines.values():
            spine.set_color(fg_hex)
        spectrum_toolbar.setStyleSheet(f"color: {fg_hex};")
        recolor_toolbar_icons()

    def disable_toolbar_navigation_mode():
        mode = str(getattr(spectrum_toolbar, "mode", ""))
        if "zoom" in mode.lower():
            spectrum_toolbar.zoom()
        elif "pan" in mode.lower():
            spectrum_toolbar.pan()

    def on_pick_mz_toggled(checked: bool):
        if checked:
            disable_toolbar_navigation_mode()

    def on_toolbar_nav_triggered(_checked=False):
        if bool(getattr(spectrum_toolbar, "mode", "")):
            pick_mz_action.setChecked(False)
        schedule_toolbar_recolor()

    pick_mz_action.toggled.connect(on_pick_mz_toggled)
    for action in spectrum_toolbar.actions():
        text = str(action.text()).lower()
        if "pan" in text or "zoom" in text:
            action.triggered.connect(on_toolbar_nav_triggered)
        action.changed.connect(schedule_toolbar_recolor)
        action.triggered.connect(schedule_toolbar_recolor)
    spectrum_canvas.mpl_connect("button_release_event", lambda event: clamp_spectrum_ylim())

    def redraw_spectrum_for_active_dataset():
        nonlocal current_mz_line
        state = get_active_state()
        coreg_dataset = state["dataset"]
        spectrum_ax.clear()
        apply_spectrum_theme()
        spectrum_ax.vlines(coreg_dataset.mz_values, 0, coreg_dataset.avg_spectrum, color="#ffffff", linewidth=0.7, alpha=0.9)
        spectrum_ax.set_xlabel("m/z")
        spectrum_ax.set_ylabel("Average intensity")
        spectrum_ax.set_title(f"Average spectrum: {state['label']}")
        current_mz_line = spectrum_ax.axvline(
            coreg_dataset.mz_values[state["current_feature_idx"]],
            color="#d7191c",
            linewidth=1.2,
            alpha=0.9,
        )
        spectrum_ax.set_ylim(bottom=0.0)
        spectrum_canvas.draw_idle()

    def on_spectrum_click(event):
        if event.xdata is None or event.inaxes is not spectrum_ax or not pick_mz_action.isChecked() or bool(getattr(spectrum_toolbar, "mode", "")):
            return
        state = get_active_state()
        coreg_dataset = state["dataset"]
        x0, x1 = spectrum_ax.get_xlim()
        x_span = abs(float(x1) - float(x0))
        half_window = max(1e-9, 0.0075 * x_span)
        idx = coreg_dataset.find_local_max_idx_near_mz(
            float(event.xdata),
            mz_window=(float(event.xdata) - half_window, float(event.xdata) + half_window),
        )
        mz_selector.target_mz.value = f"{float(coreg_dataset.mz_values[idx]):.4f}"
        update_ion_view_for_mz(float(coreg_dataset.mz_values[idx]), float(state["current_ppm_tolerance"]))

    spectrum_canvas.mpl_connect("button_press_event", on_spectrum_click)
    spectrum_ax.callbacks.connect("ylim_changed", clamp_spectrum_ylim)
    apply_spectrum_theme()

    @magicgui(
        normalize_to_tic={"widget_type": "CheckBox", "text": "Normalize to TIC"},
        contrast_percentiles={
            "widget_type": "FloatRangeSlider",
            "min": 0.0,
            "max": 100.0,
            "step": 0.1,
            "label": "Contrast percentiles",
        },
        auto_call=True,
    )
    def ion_display_options(
        normalize_to_tic=True,
        contrast_percentiles: tuple[float, float] = (1.0, 99.5),
    ):
        state = get_active_state()
        state["current_normalize_to_tic"] = bool(normalize_to_tic)
        low, high = contrast_percentiles
        low = float(low)
        high = float(high)
        if high <= low:
            high = min(100.0, low + 0.1)
            ion_display_options.contrast_percentiles.value = (low, high)
        state["current_contrast_low_pct"] = low
        state["current_contrast_high_pct"] = high
        update_ion_view_for_mz(state["current_target_mz"], state["current_ppm_tolerance"])

    @magicgui(
        target_mz={"widget_type": "LineEdit"},
        ppm_tolerance={"widget_type": "FloatSpinBox", "min": 0.1, "step": 0.5},
        auto_call=True,
    )
    def mz_selector(target_mz: str = f"{float(initial_state['dataset'].mz_values[initial_state['current_feature_idx']]):.4f}", ppm_tolerance: float = 5.0):
        try:
            parsed_target_mz = float(str(target_mz).strip())
        except Exception:
            return
        update_ion_view_for_mz(parsed_target_mz, float(ppm_tolerance))

    @magicgui(
        show_mask={"widget_type": "CheckBox", "text": "Show ROI selection mask"},
        roi_shape_key={"widget_type": "ComboBox", "choices": ["(none)"]},
        region_label={"widget_type": "ComboBox", "choices": ["(all regions)"]},
        auto_call=True,
    )
    def roi_mask_controls(show_mask: bool = False, roi_shape_key: str = "(none)", region_label: str = "(all regions)"):
        state = get_active_state()
        coreg_dataset = state["dataset"]
        if not show_mask or roi_shape_key not in coreg_dataset.sdata.shapes:
            roi_mask_layer = state.get("roi_mask_layer")
            if roi_mask_layer is not None:
                roi_mask_layer.visible = False
            return
        roi_mask_layer = ensure_roi_mask_layer(state)
        choices = annotation_region_choices(coreg_dataset, roi_shape_key)
        roi_mask_controls.region_label.choices = choices
        if roi_mask_controls.region_label.value not in choices:
            roi_mask_controls.region_label.value = choices[0]
            region_label = choices[0]
        selected = compute_annotation_region_mask(state, roi_shape_key, region_label)
        mask = np.zeros((coreg_dataset.ny, coreg_dataset.nx), dtype=np.uint8)
        mask[coreg_dataset.y_coords[selected], coreg_dataset.x_coords[selected]] = 1
        roi_mask_layer.data = mask
        roi_mask_layer.visible = True

    @magicgui(
        edge_width={"widget_type": "FloatSlider", "min": 1.0, "max": 20.0, "step": 0.5},
        edge_opacity={"widget_type": "FloatSlider", "min": 0.1, "max": 1.0, "step": 0.05},
        show_labels={"widget_type": "CheckBox", "text": "Show annotation labels"},
        label_size={"widget_type": "SpinBox", "min": 6, "max": 32, "step": 1},
        auto_call=True,
    )
    def annotation_display_controls(
        edge_width: float = 1.5,
        edge_opacity: float = 0.9,
        show_labels: bool = False,
        label_size: int = 10,
    ):
        nonlocal annotation_edge_width, annotation_edge_opacity, annotation_show_labels, annotation_label_size
        annotation_edge_width = float(edge_width)
        annotation_edge_opacity = float(edge_opacity)
        annotation_show_labels = bool(show_labels)
        annotation_label_size = int(label_size)
        apply_annotation_visuals()

    @magicgui(
        normalize_to_tic={"widget_type": "CheckBox", "text": "Normalize to TIC"},
        call_button="Export MSI From ROI",
    )
    def export_msi_from_roi_widget(normalize_to_tic: bool = True):
        state = get_active_state()
        coreg_dataset = state["dataset"]
        roi_shape_key = str(roi_mask_controls.roi_shape_key.value)
        region_label = str(roi_mask_controls.region_label.value)
        if roi_shape_key == "(none)" or roi_shape_key not in coreg_dataset.sdata.shapes:
            QMessageBox.warning(None, "Export MSI From ROI", "Select an annotation layer in ROI selection first.")
            return
        selected = compute_annotation_region_mask(state, roi_shape_key, region_label)
        if not np.any(selected):
            QMessageBox.warning(None, "Export MSI From ROI", "No MSI spectra fall inside the selected region.")
            return
        summary = coreg_dataset.summarize_region_spectra(selected, normalize_to_tic=bool(normalize_to_tic))
        default_name = sanitize_name(f"{state['label']}_{roi_shape_key}_{region_label or 'all_regions'}_msi_summary") or "msi_region_summary"
        dialog = QFileDialog(None, "Export MSI summary from ROI")
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setFileMode(QFileDialog.FileMode.AnyFile)
        dialog.setNameFilter("CSV (*.csv)")
        dialog.setDirectory(str(Path(coreg_dataset.zarr_path).parent))
        dialog.selectFile(f"{default_name}.csv")
        if dialog.exec() != QFileDialog.DialogCode.Accepted:
            return
        selected_files = dialog.selectedFiles()
        if not selected_files:
            return
        path = Path(selected_files[0]).expanduser()
        if path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")
        rows = np.column_stack(
            [
                np.asarray(summary["mz"], dtype=float),
                np.asarray(summary["mean_intensity"], dtype=float),
                np.asarray(summary["std_intensity"], dtype=float),
                np.full_like(np.asarray(summary["mz"], dtype=float), float(summary["n_spectra"]), dtype=float),
            ]
        )
        np.savetxt(path, rows, delimiter=",", header="mz,average_intensity,standard_deviation,n_spectra", comments="")

    @magicgui(
        include_same_label={"widget_type": "CheckBox", "text": "Include all regions with the same label"},
        normalize_to_tic={"widget_type": "CheckBox", "text": "Normalize to TIC"},
        call_button="Export MSI From Selected Annotation(s)",
    )
    def export_selected_annotations_widget(include_same_label: bool = False, normalize_to_tic: bool = True):
        active_layer = getattr(viewer.layers.selection, "active", None)
        source_state, shape_key, selected_mask, default_label = compute_selected_annotation_mask(
            active_layer,
            include_same_label=bool(include_same_label),
        )
        if source_state is None or shape_key is None or selected_mask is None:
            QMessageBox.warning(
                None,
                "Export MSI From Selected Annotation(s)",
                "Select an annotation shapes layer and click one or more shapes first.",
            )
            return
        if not np.any(selected_mask):
            QMessageBox.warning(
                None,
                "Export MSI From Selected Annotation(s)",
                "No MSI spectra fall inside the selected annotation region(s).",
            )
            return
        coreg_dataset = source_state["dataset"]
        summary = coreg_dataset.summarize_region_spectra(selected_mask, normalize_to_tic=bool(normalize_to_tic))
        default_name = sanitize_name(f"{source_state['label']}_{shape_key}_{default_label or 'selected_annotations'}_msi_summary") or "selected_annotation_msi_summary"
        dialog = QFileDialog(None, "Export MSI from selected annotations")
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setFileMode(QFileDialog.FileMode.AnyFile)
        dialog.setNameFilter("CSV (*.csv)")
        dialog.setDirectory(str(Path(coreg_dataset.zarr_path).parent))
        dialog.selectFile(f"{default_name}.csv")
        if dialog.exec() != QFileDialog.DialogCode.Accepted:
            return
        selected_files = dialog.selectedFiles()
        if not selected_files:
            return
        path = Path(selected_files[0]).expanduser()
        if path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")
        rows = np.column_stack(
            [
                np.asarray(summary["mz"], dtype=float),
                np.asarray(summary["mean_intensity"], dtype=float),
                np.asarray(summary["std_intensity"], dtype=float),
                np.full_like(np.asarray(summary["mz"], dtype=float), float(summary["n_spectra"]), dtype=float),
            ]
        )
        np.savetxt(path, rows, delimiter=",", header="mz,average_intensity,standard_deviation,n_spectra", comments="")

    @magicgui(
        show_pixels={"widget_type": "CheckBox", "text": "Show MSI pixels in selected annotation(s)"},
        include_same_label={"widget_type": "CheckBox", "text": "Include all regions with the same label"},
        auto_call=True,
    )
    def view_selected_annotation_pixels_widget(show_pixels: bool = False, include_same_label: bool = False):
        active_layer = getattr(viewer.layers.selection, "active", None)
        if not bool(show_pixels):
            state = get_active_state()
            selected_mask_layer = state.get("selected_annotation_mask_layer")
            if selected_mask_layer is not None:
                selected_mask_layer.visible = False
            return
        source_state, _shape_key, selected_mask, _default_label = compute_selected_annotation_mask(
            active_layer,
            include_same_label=bool(include_same_label),
        )
        if source_state is None or selected_mask is None:
            state = get_active_state()
            selected_mask_layer = state.get("selected_annotation_mask_layer")
            if selected_mask_layer is not None:
                selected_mask_layer.visible = False
            return
        if not np.any(selected_mask):
            selected_mask_layer = source_state.get("selected_annotation_mask_layer")
            if selected_mask_layer is not None:
                selected_mask_layer.visible = False
            return
        coreg_dataset = source_state["dataset"]
        mask = np.zeros((coreg_dataset.ny, coreg_dataset.nx), dtype=np.uint8)
        mask[coreg_dataset.y_coords[selected_mask], coreg_dataset.x_coords[selected_mask]] = 1
        selected_mask_layer = ensure_selected_annotation_mask_layer(source_state)
        selected_mask_layer.data = mask
        selected_mask_layer.visible = True
        if str(source_state["id"]) == str(active_dataset_label):
            try:
                viewer.layers.selection.active = active_layer
            except Exception:
                pass

    ref_landmarks = viewer.add_points(name="Reference landmarks", ndim=2, face_color="#1f77b4", size=8)
    if hasattr(ref_landmarks, "border_color"):
        ref_landmarks.border_color = "white"
    if hasattr(ref_landmarks, "editable"):
        ref_landmarks.editable = True
    if hasattr(ref_landmarks, "mode"):
        ref_landmarks.mode = "add"

    def update_landmark_numbering(layer, base_name: str, text_color: str):
        n = int(np.asarray(layer.data).shape[0])
        layer.name = f"{base_name} (next {n + 1})"
        if n == 0:
            try:
                layer.properties = {"idx": np.array([], dtype=str)}
                layer.text = None
            except Exception:
                pass
            return

        idx_labels = np.arange(1, n + 1).astype(str)
        try:
            layer.properties = {"idx": idx_labels}
            layer.text = {
                "string": "{idx}",
                "size": 11,
                "color": text_color,
                "anchor": "center",
            }
        except Exception:
            pass

    def refresh_all_landmark_numbering(_event=None):
        for label, state in datasets.items():
            update_landmark_numbering(state["msi_landmarks"], f"{label} MSI landmarks", "#ff7f0e")
        update_landmark_numbering(ref_landmarks, "Reference landmarks", "#1f77b4")

    try:
        ref_landmarks.events.data.connect(refresh_all_landmark_numbering)
    except Exception:
        pass
    for state in datasets.values():
        try:
            state["msi_landmarks"].events.data.connect(refresh_all_landmark_numbering)
        except Exception:
            pass
    refresh_all_landmark_numbering()

    def affine_from_point_pairs(src_xy: np.ndarray, dst_xy: np.ndarray) -> np.ndarray:
        a = np.column_stack([src_xy[:, 0], src_xy[:, 1], np.ones(src_xy.shape[0])])
        px, *_ = np.linalg.lstsq(a, dst_xy[:, 0], rcond=None)
        py, *_ = np.linalg.lstsq(a, dst_xy[:, 1], rcond=None)
        return np.array([[px[0], px[1], px[2]], [py[0], py[1], py[2]], [0.0, 0.0, 1.0]], dtype=float)

    def apply_linear_about_current_center(linear_xy: np.ndarray):
        state = get_active_state()
        coreg_dataset = state["dataset"]
        src_center = np.array([[(coreg_dataset.nx - 1) / 2.0], [(coreg_dataset.ny - 1) / 2.0], [1.0]], dtype=float)
        tgt_center = state["current_transform_xy"] @ src_center
        cx = float(tgt_center[0, 0])
        cy = float(tgt_center[1, 0])
        to_origin = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]], dtype=float)
        back = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]], dtype=float)
        state["current_transform_xy"][:] = (back @ linear_xy @ to_origin) @ state["current_transform_xy"]
        apply_transform_to_state(state)
        sync_controls_to_active_dataset()

    @magicgui(call_button="Fit affine from landmarks")
    def fit_affine_from_landmarks():
        state = get_active_state()
        msi_pts = np.asarray(state["msi_landmarks"].data, dtype=float)
        ref_pts = np.asarray(ref_landmarks.data, dtype=float)
        if msi_pts.shape[0] != ref_pts.shape[0] or msi_pts.shape[0] < 3:
            return
        src_xy = np.column_stack([msi_pts[:, 1], msi_pts[:, 0]])
        dst_xy = np.column_stack([ref_pts[:, 1], ref_pts[:, 0]])
        state["current_transform_xy"][:] = affine_from_point_pairs(src_xy, dst_xy) @ state["current_transform_xy"]
        apply_transform_to_state(state)
        sync_controls_to_active_dataset()

    @magicgui(call_button="Rotate MSI 180°")
    def rotate_180():
        apply_linear_about_current_center(np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float))

    @magicgui(call_button="Rotate MSI 90° CW")
    def rotate_90_cw():
        apply_linear_about_current_center(np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float))

    @magicgui(call_button="Rotate MSI 90° CCW")
    def rotate_90_ccw():
        apply_linear_about_current_center(np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float))

    @magicgui(call_button="Flip MSI horizontally")
    def flip_horizontal():
        apply_linear_about_current_center(np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float))

    @magicgui(call_button="Flip MSI vertically")
    def flip_vertical():
        apply_linear_about_current_center(np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float))

    @magicgui(call_button="Clear landmarks")
    def clear_landmarks():
        get_active_state()["msi_landmarks"].data = np.empty((0, 2), dtype=float)
        ref_landmarks.data = np.empty((0, 2), dtype=float)
        refresh_all_landmark_numbering()

    def activate_landmark_picking(target: str):
        state = get_active_state()
        layer = state["msi_landmarks"] if str(target).lower() == "msi" else ref_landmarks
        layer.visible = True
        try:
            viewer.layers.selection.active = layer
        except Exception:
            pass
        if hasattr(layer, "editable"):
            layer.editable = True
        if hasattr(layer, "mode"):
            try:
                layer.mode = "add"
            except Exception:
                pass

    def stop_landmark_picking():
        for layer in [ref_landmarks, *[state["msi_landmarks"] for state in datasets.values()]]:
            if hasattr(layer, "mode"):
                try:
                    layer.mode = "select"
                except Exception:
                    pass

    @magicgui(call_button="Pick MSI Points")
    def pick_msi_landmarks_widget():
        activate_landmark_picking("msi")

    @magicgui(call_button="Pick Reference Points")
    def pick_reference_landmarks_widget():
        activate_landmark_picking("reference")

    @magicgui(call_button="Stop Picking")
    def stop_landmark_picking_widget():
        stop_landmark_picking()

    initial_dataset_choices = current_dataset_choices()
    initial_active_choice = dataset_choice_text(initial_state)

    @magicgui(display_name={"widget_type": "LineEdit"}, call_button="Rename Active Dataset")
    def rename_dataset_widget(display_name: str = str(initial_state["label"])):
        state = get_active_state()
        new_name = rename_msi_dataset(state["dataset"].zarr_path, table_key=state["dataset"].table_key, display_name=display_name)
        state["label"] = new_name
        state["dataset"].display_name = new_name
        roi_mask_layer = state.get("roi_mask_layer")
        if roi_mask_layer is not None:
            roi_mask_layer.name = f"{new_name} ROI selection mask"
        selected_annotation_mask_layer = state.get("selected_annotation_mask_layer")
        if selected_annotation_mask_layer is not None:
            selected_annotation_mask_layer.name = f"{new_name} selected annotation mask"
        state["msi_landmarks"].name = f"{new_name} MSI landmarks"
        update_ion_view(state["current_feature_idx"])
        refresh_all_landmark_numbering()
        sync_controls_to_active_dataset()

    initial_target_choices = [choice for choice in initial_dataset_choices if choice != initial_active_choice]

    @magicgui(
        target_dataset={"widget_type": "ComboBox", "choices": initial_target_choices if initial_target_choices else [""]},
        call_button="Copy Active Affine To Target",
    )
    def copy_affine_to_target_widget(target_dataset: str = (initial_target_choices[0] if initial_target_choices else "")):
        source_state = get_active_state()
        target_key = dataset_choice_to_key.get(str(target_dataset))
        if target_key is None or target_key not in datasets or target_key == source_state["id"]:
            return
        target_state = datasets[target_key]
        target_state["current_transform_xy"][:] = np.asarray(source_state["current_transform_xy"], dtype=float)
        apply_transform_to_state(target_state)
        if str(target_state["id"]) == str(active_dataset_label):
            sync_controls_to_active_dataset()

    msi_layer_controls = QWidget()
    msi_layer_controls_layout = QGridLayout(msi_layer_controls)
    msi_layer_controls_layout.setContentsMargins(0, 0, 0, 0)
    msi_layer_controls_layout.setHorizontalSpacing(4)
    msi_layer_controls_layout.setVerticalSpacing(2)
    msi_layer_active_group = QButtonGroup(msi_layer_controls)
    msi_layer_active_group.setExclusive(True)

    def rebuild_msi_layer_controls():
        nonlocal msi_layer_active_group
        clear_layout(msi_layer_controls_layout)
        msi_layer_active_group = QButtonGroup(msi_layer_controls)
        msi_layer_active_group.setExclusive(True)
        msi_layer_controls_layout.addWidget(QLabel("Dataset"), 0, 0)
        msi_layer_controls_layout.addWidget(QLabel("Active"), 0, 1)
        msi_layer_controls_layout.addWidget(QLabel("Show"), 0, 2)
        msi_layer_controls_layout.addWidget(QLabel("Cmap"), 0, 3)

        for row_idx, state in enumerate(datasets.values(), start=1):
            dataset_key = str(state["id"])
            label = QLabel(str(state["label"]))
            active_button = QRadioButton()
            active_button.setChecked(dataset_key == str(active_dataset_label))
            checkbox = QCheckBox()
            checkbox.setChecked(bool(state["ion_layer"].visible))
            cmap_combo = QComboBox()
            cmap_combo.addItems(list(overlay_colormap_order))
            cmap_combo.setCurrentText(str(state["current_colormap_name"]))
            cmap_combo.setMaximumWidth(110)
            if hasattr(QComboBox, "SizeAdjustPolicy"):
                cmap_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContentsOnFirstShow)

            def _set_visible(checked, dataset_key=dataset_key):
                if dataset_key not in datasets:
                    return
                datasets[dataset_key]["ion_layer"].visible = bool(checked)

            def _set_active(checked, dataset_key=dataset_key):
                if checked:
                    set_active_dataset(dataset_key)

            def _set_colormap(value, dataset_key=dataset_key):
                if dataset_key not in datasets or value not in mpl_colormaps:
                    return
                datasets[dataset_key]["current_colormap_name"] = str(value)
                datasets[dataset_key]["ion_layer"].colormap = get_overlay_colormap(str(value))

            msi_layer_active_group.addButton(active_button)
            active_button.toggled.connect(_set_active)
            checkbox.toggled.connect(_set_visible)
            cmap_combo.currentTextChanged.connect(_set_colormap)

            msi_layer_controls_layout.addWidget(label, row_idx, 0)
            msi_layer_controls_layout.addWidget(active_button, row_idx, 1)
            msi_layer_controls_layout.addWidget(checkbox, row_idx, 2)
            msi_layer_controls_layout.addWidget(cmap_combo, row_idx, 3)

    if_layer_controls = QWidget()
    if_layer_controls_layout = QGridLayout(if_layer_controls)
    if_layer_controls_layout.setContentsMargins(0, 0, 0, 0)
    if_layer_controls_layout.setHorizontalSpacing(4)
    if_layer_controls_layout.setVerticalSpacing(2)

    def rebuild_if_layer_controls():
        clear_layout(if_layer_controls_layout)
        if_layer_controls_layout.addWidget(QLabel("IF channel"), 0, 0)
        if_layer_controls_layout.addWidget(QLabel("Show"), 0, 1)
        if_layer_controls_layout.addWidget(QLabel("Opacity"), 0, 2)
        if_layer_controls_layout.addWidget(QLabel("Color"), 0, 3)
        if_layer_controls_layout.addWidget(QLabel("Solo"), 0, 4)

        layers = _reference_channel_layers()
        if not layers:
            if_layer_controls_layout.addWidget(QLabel("No IF/reference layers loaded"), 1, 0, 1, 5)
            return

        for row_idx, layer in enumerate(layers, start=1):
            metadata = _layer_metadata(layer)
            label = QLabel(str(layer.name))
            show_box = QCheckBox()
            show_box.setChecked(bool(layer.visible))
            opacity_spin = QDoubleSpinBox()
            opacity_spin.setRange(0.0, 1.0)
            opacity_spin.setSingleStep(0.05)
            opacity_spin.setDecimals(2)
            opacity_spin.setValue(float(getattr(layer, "opacity", 1.0)))
            color_combo = QComboBox()
            color_combo.addItems(_reference_color_choices())
            color_choice = str(metadata.get("reference_color_choice", "metadata"))
            color_combo.setCurrentText(color_choice if color_choice in _reference_color_choices() else "metadata")
            solo_button = QPushButton("Solo")

            def _set_layer_visible(checked, layer=layer):
                layer.visible = bool(checked)

            def _set_layer_opacity(value, layer=layer):
                layer.opacity = float(value)

            def _set_layer_color(value, layer=layer):
                _set_reference_layer_color(layer, str(value))

            def _solo_layer(_checked=False, layer=layer):
                for other in _reference_channel_layers():
                    other.visible = other is layer
                rebuild_if_layer_controls()

            show_box.toggled.connect(_set_layer_visible)
            opacity_spin.valueChanged.connect(_set_layer_opacity)
            color_combo.currentTextChanged.connect(_set_layer_color)
            solo_button.clicked.connect(_solo_layer)

            if_layer_controls_layout.addWidget(label, row_idx, 0)
            if_layer_controls_layout.addWidget(show_box, row_idx, 1)
            if_layer_controls_layout.addWidget(opacity_spin, row_idx, 2)
            if_layer_controls_layout.addWidget(color_combo, row_idx, 3)
            if_layer_controls_layout.addWidget(solo_button, row_idx, 4)

    @magicgui(
        layer_name={"widget_type": "ComboBox", "choices": _reference_channel_choice_names()},
        visible={"widget_type": "CheckBox", "text": "Visible"},
        display_name={"widget_type": "LineEdit"},
        blending={"widget_type": "ComboBox", "choices": ["translucent", "additive", "opaque"]},
        color={"widget_type": "ComboBox", "choices": _reference_color_choices()},
        contrast_percentiles={
            "widget_type": "FloatRangeSlider",
            "min": 0.0,
            "max": 100.0,
            "step": 0.1,
            "label": "Contrast percentiles",
        },
        call_button="Apply IF Channel Settings",
    )
    def if_channel_editor(
        layer_name: str = _reference_channel_choice_names()[0],
        visible: bool = True,
        display_name: str = "",
        blending: str = "translucent",
        color: str = "metadata",
        contrast_percentiles: tuple[float, float] = (1.0, 99.8),
    ):
        layer = _get_reference_layer_by_name(layer_name)
        if layer is None:
            return
        layer.visible = bool(visible)
        layer.blending = str(blending)
        new_name = str(display_name).strip()
        if new_name:
            layer.name = new_name
        _set_reference_layer_color(layer, str(color))
        low_pct = float(contrast_percentiles[0])
        high_pct = float(contrast_percentiles[1])
        if high_pct <= low_pct:
            high_pct = min(100.0, low_pct + 0.1)
        try:
            layer.contrast_limits = auto_contrast_limits(np.asarray(layer.data), low_pct=low_pct, high_pct=high_pct)
        except Exception:
            pass
        _refresh_if_toolbox_widgets(preferred_layer_name=str(layer.name))

    @magicgui(
        layer_name={"widget_type": "ComboBox", "choices": _reference_channel_choice_names()},
        show_all={"widget_type": "CheckBox", "text": "Show all IF channels"},
        opacity={"widget_type": "FloatSlider", "min": 0.0, "max": 1.0, "step": 0.05},
        auto_call=True,
    )
    def if_channel_visibility_widget(layer_name: str = _reference_channel_choice_names()[0], show_all: bool = False, opacity: float = 1.0):
        selected = _get_reference_layer_by_name(layer_name)
        for layer in _reference_channel_layers():
            if bool(show_all):
                layer.visible = True
            elif selected is not None:
                layer.visible = layer is selected or bool(layer.visible and layer is not selected)
        if selected is not None:
            selected.opacity = float(opacity)
            metadata = _layer_metadata(selected)
            if_channel_editor["visible"].value = bool(selected.visible)
            if_channel_editor["display_name"].value = str(selected.name)
            if_channel_editor["blending"].value = str(getattr(selected, "blending", "translucent"))
            if_channel_editor["color"].value = str(metadata.get("reference_color_choice", "metadata"))
        rebuild_if_layer_controls()

    def sync_controls_to_active_dataset():
        state = get_active_state()
        coreg_dataset = state["dataset"]
        dataset_choice_to_key.clear()
        ordered_choices = []
        for item in datasets.values():
            choice = dataset_choice_text(item)
            dataset_choice_to_key[choice] = str(item["id"])
            ordered_choices.append(choice)
        copy_affine_to_target_widget.target_dataset.choices = ordered_choices
        rename_dataset_widget.display_name.value = str(state["label"])
        target_choices = [choice for choice in ordered_choices if dataset_choice_to_key.get(choice) != str(state["id"])]
        copy_affine_to_target_widget.target_dataset.choices = target_choices if target_choices else [""]
        if copy_affine_to_target_widget.target_dataset.value not in copy_affine_to_target_widget.target_dataset.choices:
            copy_affine_to_target_widget.target_dataset.value = copy_affine_to_target_widget.target_dataset.choices[0]
        ion_display_options.normalize_to_tic.value = bool(state["current_normalize_to_tic"])
        ion_display_options.contrast_percentiles.value = (
            float(state["current_contrast_low_pct"]),
            float(state["current_contrast_high_pct"]),
        )
        mz_selector.target_mz.value = f"{float(state['current_target_mz']):.4f}"
        mz_selector.ppm_tolerance.value = float(state["current_ppm_tolerance"])
        roi_shape_keys = [key for key in coreg_dataset.sdata.shapes.keys() if "pixels" not in key.lower()]
        roi_mask_controls.roi_shape_key.choices = roi_shape_keys if roi_shape_keys else ["(none)"]
        if roi_mask_controls.roi_shape_key.value not in roi_mask_controls.roi_shape_key.choices:
            roi_mask_controls.roi_shape_key.value = roi_mask_controls.roi_shape_key.choices[0]
        roi_label_choices = annotation_region_choices(coreg_dataset, str(roi_mask_controls.roi_shape_key.value))
        roi_mask_controls.region_label.choices = roi_label_choices
        if roi_mask_controls.region_label.value not in roi_label_choices:
            roi_mask_controls.region_label.value = roi_label_choices[0]
        refresh_annotation_widget_choices()
        rebuild_msi_layer_controls()
        _refresh_if_toolbox_widgets()
        try:
            alignment_active_dataset_label.setText(f"Active dataset: {state['label']}")
        except Exception:
            pass
        for key, other_state in datasets.items():
            other_state["msi_landmarks"].visible = (key == str(state["id"]))
            if key != str(state["id"]):
                roi_mask_layer = other_state.get("roi_mask_layer")
                if roi_mask_layer is not None:
                    roi_mask_layer.visible = False
                selected_annotation_mask_layer = other_state.get("selected_annotation_mask_layer")
                if selected_annotation_mask_layer is not None:
                    selected_annotation_mask_layer.visible = False
        redraw_spectrum_for_active_dataset()
        try:
            roi_mask_controls()
        except Exception:
            pass

    def set_active_dataset(label: str):
        nonlocal active_dataset_label
        if label not in datasets:
            return
        active_dataset_label = label
        sync_controls_to_active_dataset()

    @magicgui(call_button="Add MSI Dataset")
    def add_msi_dataset():
        picked_path = _pick_input_or_convert()
        embedded = embed_msi_dataset(host_zarr_path, picked_path, registered_cs=registered_cs)
        new_dataset = CoregistrationDataset(
            host_zarr_path,
            registered_cs=registered_cs,
            table_key=embedded["table_key"],
            tic_key=embedded["tic_key"],
        )
        state = add_dataset_to_view(new_dataset, str(new_dataset.display_name))
        enforce_reference_layers_at_bottom()
        add_annotation_shape_layers(state)
        try:
            state["msi_landmarks"].events.data.connect(refresh_all_landmark_numbering)
        except Exception:
            pass
        refresh_all_landmark_numbering()
        state["ion_layer"].visible = True
        set_active_dataset(str(state["id"]))

    @magicgui(call_button="Add/Update Optical")
    def add_optical_image():
        state = get_active_state()
        coreg_dataset = state["dataset"]
        path, _ = QFileDialog.getOpenFileName(None, "Select optical image", "", "Image files (*.tif *.tiff *.png *.jpg *.jpeg);;All files (*)")
        if not path:
            return
        add_reference_image(coreg_dataset.zarr_path, path, key="optical", registered_cs=registered_cs)
        coreg_dataset.sdata = sd.read_zarr(coreg_dataset.zarr_path)
        if "optical" not in coreg_dataset.reference_image_keys:
            coreg_dataset.reference_image_keys.append("optical")
        add_or_update_reference_layer(coreg_dataset, "optical")

    @magicgui(call_button="Add/Update H&E")
    def add_hne_image():
        state = get_active_state()
        coreg_dataset = state["dataset"]
        path, _ = QFileDialog.getOpenFileName(None, "Select H&E image", "", "Image files (*.tif *.tiff *.png *.jpg *.jpeg);;All files (*)")
        if not path:
            return
        add_reference_image(coreg_dataset.zarr_path, path, key="hne", registered_cs=registered_cs)
        coreg_dataset.sdata = sd.read_zarr(coreg_dataset.zarr_path)
        if "hne" not in coreg_dataset.reference_image_keys:
            coreg_dataset.reference_image_keys.append("hne")
        add_or_update_reference_layer(coreg_dataset, "hne")

    @magicgui(
        qptiff_level={"widget_type": "SpinBox", "min": 0, "max": 12, "step": 1},
        call_button="Add/Update H&E From QPTIFF",
    )
    def add_hne_from_qptiff(qptiff_level: int = 0):
        state = get_active_state()
        coreg_dataset = state["dataset"]
        path, _ = QFileDialog.getOpenFileName(None, "Select H&E QPTIFF", "", "QPTIFF/OME-TIFF (*.qptiff *.ome.tif *.ome.tiff *.tif *.tiff);;All files (*)")
        if not path:
            return
        add_reference_image(coreg_dataset.zarr_path, path, key="hne", registered_cs=registered_cs, qptiff_level=int(qptiff_level))
        coreg_dataset.sdata = sd.read_zarr(coreg_dataset.zarr_path)
        if "hne" not in coreg_dataset.reference_image_keys:
            coreg_dataset.reference_image_keys.append("hne")
        add_or_update_reference_layer(coreg_dataset, "hne")

    @magicgui(
        target_image={"widget_type": "ComboBox", "choices": ["hne", "optical"]},
        name_prefix={"widget_type": "LineEdit"},
        object_mode={"widget_type": "ComboBox", "choices": ["annotations_only", "non_cell", "all", "cells_only"]},
        max_shapes={"widget_type": "SpinBox", "min": 0, "max": 1000000, "step": 1000},
        simplify_tolerance={"widget_type": "FloatSpinBox", "min": 0.0, "max": 1000.0, "step": 0.5},
        annotation_pyramid_level={"widget_type": "SpinBox", "min": -1, "max": 12, "step": 1},
        call_button="Add GeoJSON",
    )
    def add_geojson_annotations(
        target_image: str = "hne",
        name_prefix: str = "anno_",
        object_mode: str = "annotations_only",
        max_shapes: int = 0,
        simplify_tolerance: float = 0.0,
        annotation_pyramid_level: int = -1,
    ):
        state = get_active_state()
        coreg_dataset = state["dataset"]
        paths, _ = QFileDialog.getOpenFileNames(None, "Select GeoJSON annotation file(s)", "", "GeoJSON (*.geojson);;All files (*)")
        if not paths:
            return
        keys = import_geojson_annotations(
            coreg_dataset.zarr_path,
            paths,
            target_image=target_image,
            name_prefix=name_prefix,
            registered_cs=registered_cs,
            object_mode=object_mode,
            max_shapes=int(max_shapes),
            simplify_tolerance=float(simplify_tolerance),
            annotation_pyramid_level=(int(annotation_pyramid_level) if int(annotation_pyramid_level) >= 0 else None),
        )
        coreg_dataset.sdata = sd.read_zarr(coreg_dataset.zarr_path)
        add_annotation_shape_layers(state, keys)
        sync_controls_to_active_dataset()

    @magicgui(
        annotation_key={"widget_type": "ComboBox", "choices": ["(none)"]},
        call_button="Remove GeoJSON",
    )
    def remove_geojson_annotations(annotation_key: str = "(none)"):
        if annotation_key == "(none)":
            return
        state = get_active_state()
        deleted = delete_geojson_annotations(state["dataset"].zarr_path, [annotation_key])
        if not deleted:
            return
        for dataset_state in datasets.values():
            dataset_state["dataset"].sdata = sd.read_zarr(dataset_state["dataset"].zarr_path)
        remove_annotation_shape_layers(deleted)
        sync_controls_to_active_dataset()

    @magicgui(
        annotation_key={"widget_type": "ComboBox", "choices": ["(none)"]},
        annotation_scale_x={"widget_type": "FloatSpinBox", "min": 0.000001, "max": 1000000.0, "step": 0.01},
        annotation_scale_y={"widget_type": "FloatSpinBox", "min": 0.000001, "max": 1000000.0, "step": 0.01},
        annotation_translate_x={"widget_type": "FloatSpinBox", "min": -1000000.0, "max": 1000000.0, "step": 1.0},
        annotation_translate_y={"widget_type": "FloatSpinBox", "min": -1000000.0, "max": 1000000.0, "step": 1.0},
        call_button="Adjust GeoJSON",
    )
    def rescale_geojson_annotations_widget(
        annotation_key: str = "(none)",
        annotation_scale_x: float = 1.0,
        annotation_scale_y: float = 1.0,
        annotation_translate_x: float = 0.0,
        annotation_translate_y: float = 0.0,
    ):
        if annotation_key == "(none)":
            return
        sx = float(annotation_scale_x)
        sy = float(annotation_scale_y)
        tx = float(annotation_translate_x)
        ty = float(annotation_translate_y)
        if np.isclose(sx, 1.0) and np.isclose(sy, 1.0) and np.isclose(tx, 0.0) and np.isclose(ty, 0.0):
            return
        state = get_active_state()
        rewritten = transform_geojson_annotations(
            state["dataset"].zarr_path,
            [annotation_key],
            annotation_scale_x=sx,
            annotation_scale_y=sy,
            annotation_translate_x=tx,
            annotation_translate_y=ty,
        )
        if not rewritten:
            return
        for dataset_state in datasets.values():
            dataset_state["dataset"].sdata = sd.read_zarr(dataset_state["dataset"].zarr_path)
        remove_annotation_shape_layers(rewritten)
        for dataset_state in datasets.values():
            add_annotation_shape_layers(dataset_state, rewritten)
        sync_controls_to_active_dataset()

    @magicgui(call_button="Save Active Registration")
    def save_registration_widget():
        state = get_active_state()
        save_coregistration(
            state["dataset"].zarr_path,
            state["current_transform_xy"],
            table_key=state["dataset"].table_key,
            tic_key=state["dataset"].tic_key,
            registered_cs=registered_cs,
        )

    @magicgui(call_button="Save All Registrations")
    def save_all_registrations_widget():
        for state in datasets.values():
            save_coregistration(
                state["dataset"].zarr_path,
                state["current_transform_xy"],
                table_key=state["dataset"].table_key,
                tic_key=state["dataset"].tic_key,
                registered_cs=registered_cs,
            )

    @magicgui(call_button="Export Current View TIFF")
    def export_current_view_widget():
        default_name = f"{get_active_state()['label']}_view.tif"
        dialog = QFileDialog(None, "Export current MSI view")
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setFileMode(QFileDialog.FileMode.AnyFile)
        dialog.setNameFilter("TIFF (*.tif *.tiff)")
        dialog.setDirectory(str(Path(get_active_state()["dataset"].zarr_path).parent))
        dialog.selectFile(default_name)
        if dialog.exec() != QFileDialog.DialogCode.Accepted:
            return
        selected = dialog.selectedFiles()
        if not selected:
            return
        path = Path(selected[0]).expanduser()
        if path.suffix.lower() not in {".tif", ".tiff"}:
            path = path.with_suffix(".tif")
        screenshot = viewer.screenshot(canvas_only=True, flash=False)
        iio.imwrite(path, np.asarray(screenshot))

    @magicgui(call_button="Reset Canvas View")
    def reset_canvas_view_widget():
        nonlocal startup_camera_state
        if not startup_camera_state:
            return
        try:
            viewer.camera.center = startup_camera_state["center"]
            viewer.camera.zoom = startup_camera_state["zoom"]
            if "angles" in startup_camera_state:
                viewer.camera.angles = startup_camera_state["angles"]
        except Exception:
            pass

    controls_scroll = QScrollArea()
    controls_scroll.setWidgetResizable(True)
    controls_container = QWidget()
    controls_layout = QVBoxLayout(controls_container)
    controls_layout.setContentsMargins(6, 6, 6, 6)
    controls_layout.setSpacing(8)
    controls_layout.addWidget(spectrum_widget)
    controls_layout.addWidget(reset_canvas_view_widget.native)
    controls_layout.addWidget(export_current_view_widget.native)
    controls_layout.addWidget(rename_dataset_widget.native)
    controls_layout.addWidget(copy_affine_to_target_widget.native)
    controls_layout.addWidget(msi_layer_controls)
    controls_layout.addWidget(ion_display_options.native)
    controls_layout.addWidget(mz_selector.native)
    controls_layout.addWidget(roi_mask_controls.native)
    controls_layout.addWidget(export_msi_from_roi_widget.native)
    controls_layout.addWidget(annotation_display_controls.native)
    controls_layout.addWidget(view_selected_annotation_pixels_widget.native)
    controls_layout.addWidget(export_selected_annotations_widget.native)
    controls_layout.addStretch(1)
    controls_scroll.setWidget(controls_container)

    if_launcher = QWidget()
    if_launcher_layout = QVBoxLayout(if_launcher)
    if_launcher_layout.setContentsMargins(0, 0, 0, 0)
    if_launcher_layout.setSpacing(6)
    if_button = QPushButton("Open IF Display Tools")
    if_launcher_layout.addWidget(if_button)
    if_launcher_layout.addWidget(QLabel("IF/reference channel visibility and contrast"))

    if_dialog = QDialog()
    if_dialog.setWindowTitle("IF Display Tools")
    if_dialog.setModal(False)
    if_dialog.resize(520, 760)
    if_dialog_layout = QVBoxLayout(if_dialog)
    if_dialog_layout.setContentsMargins(8, 8, 8, 8)
    if_dialog_layout.setSpacing(8)
    if_dialog_layout.addWidget(if_layer_controls)
    if_dialog_layout.addWidget(if_channel_visibility_widget.native)
    if_dialog_layout.addWidget(if_channel_editor.native)
    if_dialog_layout.addStretch(1)

    def open_if_dialog():
        if_dialog.show()
        if_dialog.raise_()
        if_dialog.activateWindow()

    if_button.clicked.connect(open_if_dialog)

    alignment_launcher = QWidget()
    alignment_launcher_layout = QVBoxLayout(alignment_launcher)
    alignment_launcher_layout.setContentsMargins(0, 0, 0, 0)
    alignment_launcher_layout.setSpacing(6)
    alignment_button = QPushButton("Open Alignment Tools")
    alignment_launcher_layout.addWidget(alignment_button)
    alignment_launcher_layout.addWidget(QLabel("Landmarks, transforms, and registration save"))

    alignment_dialog = QDialog()
    alignment_dialog.setWindowTitle("Alignment Tools")
    alignment_dialog.setModal(False)
    alignment_dialog.resize(420, 620)
    alignment_dialog_layout = QVBoxLayout(alignment_dialog)
    alignment_dialog_layout.setContentsMargins(8, 8, 8, 8)
    alignment_dialog_layout.setSpacing(8)
    alignment_active_dataset_label = QLabel("")
    alignment_dialog_layout.addWidget(alignment_active_dataset_label)
    alignment_dialog_layout.addWidget(QLabel("Landmark picking"))
    alignment_dialog_layout.addWidget(pick_msi_landmarks_widget.native)
    alignment_dialog_layout.addWidget(pick_reference_landmarks_widget.native)
    alignment_dialog_layout.addWidget(stop_landmark_picking_widget.native)
    alignment_dialog_layout.addWidget(QLabel("Alignment actions"))
    alignment_dialog_layout.addWidget(fit_affine_from_landmarks.native)
    alignment_dialog_layout.addWidget(rotate_180.native)
    alignment_dialog_layout.addWidget(rotate_90_cw.native)
    alignment_dialog_layout.addWidget(rotate_90_ccw.native)
    alignment_dialog_layout.addWidget(flip_horizontal.native)
    alignment_dialog_layout.addWidget(flip_vertical.native)
    alignment_dialog_layout.addWidget(clear_landmarks.native)
    alignment_dialog_layout.addWidget(QLabel("Registration"))
    alignment_dialog_layout.addWidget(save_registration_widget.native)
    alignment_dialog_layout.addWidget(save_all_registrations_widget.native)
    alignment_dialog_layout.addStretch(1)

    def open_alignment_dialog():
        alignment_dialog.show()
        alignment_dialog.raise_()
        alignment_dialog.activateWindow()

    alignment_button.clicked.connect(open_alignment_dialog)

    add_data_launcher = QWidget()
    add_data_launcher_layout = QVBoxLayout(add_data_launcher)
    add_data_launcher_layout.setContentsMargins(6, 6, 6, 6)
    add_data_launcher_layout.setSpacing(6)
    add_data_button = QPushButton("Open Add Data Tools")
    add_data_launcher_layout.addWidget(add_data_button)
    add_data_launcher_layout.addWidget(QLabel("Imports and annotation tools"))
    add_data_launcher_layout.addStretch(1)

    add_data_dialog = QDialog()
    add_data_dialog.setWindowTitle("Add Data")
    add_data_dialog.setModal(False)
    add_data_dialog.resize(520, 900)
    add_data_dialog_layout = QVBoxLayout(add_data_dialog)
    add_data_dialog_layout.setContentsMargins(8, 8, 8, 8)
    add_data_dialog_layout.setSpacing(8)
    add_data_dialog_scroll = QScrollArea()
    add_data_dialog_scroll.setWidgetResizable(True)
    add_data_dialog_container = QWidget()
    add_data_dialog_container_layout = QVBoxLayout(add_data_dialog_container)
    add_data_dialog_container_layout.setContentsMargins(4, 4, 4, 4)
    add_data_dialog_container_layout.setSpacing(8)
    add_data_dialog_container_layout.addWidget(add_msi_dataset.native)
    add_data_dialog_container_layout.addWidget(add_optical_image.native)
    add_data_dialog_container_layout.addWidget(add_hne_image.native)
    add_data_dialog_container_layout.addWidget(add_hne_from_qptiff.native)
    add_data_dialog_container_layout.addWidget(add_geojson_annotations.native)
    add_data_dialog_container_layout.addWidget(remove_geojson_annotations.native)
    add_data_dialog_container_layout.addWidget(rescale_geojson_annotations_widget.native)
    add_data_dialog_container_layout.addStretch(1)
    add_data_dialog_scroll.setWidget(add_data_dialog_container)
    add_data_dialog_layout.addWidget(add_data_dialog_scroll)

    def open_add_data_dialog():
        add_data_dialog.show()
        add_data_dialog.raise_()
        add_data_dialog.activateWindow()

    add_data_button.clicked.connect(open_add_data_dialog)

    viewer.window.add_dock_widget(controls_scroll, area="right", name="Controls")
    viewer.window.add_dock_widget(add_data_launcher, area="left", name="Add Data")
    viewer.window.add_dock_widget(if_launcher, area="right", name="IF Display")
    viewer.window.add_dock_widget(alignment_launcher, area="right", name="Alignment")
    enforce_reference_layers_at_bottom()
    add_annotation_shape_layers(initial_state)
    sync_controls_to_active_dataset()
    try:
        viewer.reset_view()
        startup_camera_state = {
            "center": tuple(viewer.camera.center),
            "zoom": float(viewer.camera.zoom),
            "angles": tuple(viewer.camera.angles),
        }
    except Exception:
        pass
    napari.run()
    return viewer
