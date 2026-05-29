from __future__ import annotations

import csv
import json
import sys
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
import napari
import tifffile
from magicgui import magicgui
from matplotlib import colormaps as mpl_colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.path import Path as MplPath
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
    QProgressDialog,
    QRadioButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely import affinity
from spatialdata._io import write_image, write_shapes, write_table


import numpy as np
import zarr
from zarr.errors import ZarrUserWarning

_QT_APP = None
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


def _ensure_qapplication(QApplication):
    global _QT_APP
    app = QApplication.instance()
    if app is None:
        _QT_APP = QApplication(sys.argv)
        app = _QT_APP
    return app


def _show_busy_dialog(title: str, message: str):
    app = QApplication.instance()
    dialog = QProgressDialog(message, "", 0, 0)
    dialog.setWindowTitle(title)
    dialog.setCancelButton(None)
    dialog.setWindowModality(Qt.ApplicationModal)
    dialog.setMinimumDuration(0)
    dialog.setValue(0)
    dialog.show()
    if app is not None:
        app.processEvents()
    return dialog


def _close_busy_dialog(dialog):
    if dialog is None:
        return
    dialog.close()
    dialog.deleteLater()
    app = QApplication.instance()
    if app is not None:
        app.processEvents()


def _run_with_busy_dialog(title: str, message: str, func):
    dialog = _show_busy_dialog(title, message)
    try:
        return func()
    finally:
        _close_busy_dialog(dialog)


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
    ref_centers = (transform @ centers.T).T[:, :2]
    nearest_x = np.rint(ref_centers[:, 0]).astype(int)
    nearest_y = np.rint(ref_centers[:, 1]).astype(int)

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
        poly_xy = (transform @ corners.T).T[:, :2]
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
        if 0 <= ref_x < width and 0 <= ref_y < height and np.isfinite(img[ref_y, ref_x]):
            values[idx] = float(img[ref_y, ref_x])

    return values


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

    def make_threshold_preview_colormap():
        colors = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.1, 0.35, 1.0, 0.25],
                [1.0, 0.1, 0.1, 0.35],
            ],
            dtype=float,
        )
        return Colormap(colors=colors, name="threshold_blue_red")

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
    threshold_preview_colormap = make_threshold_preview_colormap()

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

    def dataset_key_from_choice(choice: str, fallback: str | None = None) -> str | None:
        text = str(choice)
        if text in dataset_choice_to_key:
            return dataset_choice_to_key[text]
        if text in datasets:
            return text
        for state in datasets.values():
            if dataset_choice_text(state) == text:
                return str(state["id"])
        return fallback

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
        initial_contrast_limits = auto_contrast_limits(initial_img)

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
            contrast_limits=initial_contrast_limits,
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
            "current_contrast_mode": "percentile",
            "current_contrast_low_pct": 1.0,
            "current_contrast_high_pct": 99.5,
            "current_contrast_low": float(initial_contrast_limits[0]),
            "current_contrast_high": float(initial_contrast_limits[1]),
            "current_transform_xy": np.asarray(saved_xy_matrix, dtype=float).copy(),
            "ion_layer": ion_layer,
            "roi_mask_layer": None,
            "selected_annotation_mask_layer": None,
            "threshold_preview_layer": None,
            "threshold_preview_img": None,
            "threshold_preview_feature_indices": None,
            "threshold_preview_mz": None,
            "threshold_preview_ppm": None,
            "threshold_preview_normalize_to_tic": None,
            "threshold_preview_source_dataset_id": None,
            "threshold_preview_source_transform_xy": None,
            "threshold_preview_target_transform_xy": None,
            "threshold_preview_values": None,
            "threshold_preview_prefilter_shape_key": "(none)",
            "threshold_preview_prefilter_region_label": "(all regions)",
            "optimization_preview_layer": None,
            "optimization_candidate_transform_xy": None,
            "optimization_previous_ion_visible": None,
            "msi_landmarks": msi_landmarks,
        }
        return state

    def apply_ion_contrast_to_active_layer(img: np.ndarray):
        state = get_active_state()
        if str(state.get("current_contrast_mode", "percentile")).lower() == "absolute":
            low = float(state.get("current_contrast_low", 0.0))
            high = float(state.get("current_contrast_high", 1.0))
            if high <= low:
                high = low + 1e-9
            state["ion_layer"].contrast_limits = (low, high)
            return
        contrast_limits = auto_contrast_limits(
            img,
            low_pct=float(state["current_contrast_low_pct"]),
            high_pct=float(state["current_contrast_high_pct"]),
        )
        state["ion_layer"].contrast_limits = contrast_limits
        state["current_contrast_low"] = float(contrast_limits[0])
        state["current_contrast_high"] = float(contrast_limits[1])
        try:
            ion_display_options.absolute_low.value = f"{float(contrast_limits[0]):g}"
            ion_display_options.absolute_high.value = f"{float(contrast_limits[1]):g}"
        except Exception:
            pass

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
        preview_layer = state.get("threshold_preview_layer")
        if preview_layer is not None:
            preview_layer.affine = aff_yx
            preview_layer.scale = (1.0, 1.0)
            preview_layer.translate = (0.0, 0.0)

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

    def ensure_threshold_preview_layer(state: dict[str, Any]):
        if state.get("threshold_preview_layer") is not None:
            return state["threshold_preview_layer"]
        coreg_dataset = state["dataset"]
        preview_layer = viewer.add_image(
            np.zeros((coreg_dataset.ny, coreg_dataset.nx), dtype=np.uint8),
            name=f"{state['label']} threshold preview",
            colormap=threshold_preview_colormap,
            contrast_limits=(0, 2),
            interpolation2d="nearest",
            blending="translucent",
            opacity=1.0,
            visible=False,
        )
        state["threshold_preview_layer"] = preview_layer
        apply_transform_to_state(state)
        return preview_layer

    def remove_threshold_preview_layers():
        for state in datasets.values():
            preview_layer = state.get("threshold_preview_layer")
            if preview_layer is None:
                continue
            try:
                viewer.layers.remove(preview_layer)
            except Exception:
                try:
                    preview_layer.visible = False
                except Exception:
                    pass
            state["threshold_preview_layer"] = None

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

    def _get_reference_layer_by_key_channel(reference_key: str, channel_index: int):
        for layer in _reference_channel_layers():
            metadata = _layer_metadata(layer)
            if str(metadata.get("reference_key", "")) == str(reference_key) and int(metadata.get("reference_channel_index", 0)) == int(channel_index):
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

    def _apply_reference_layer_contrast(
        layer,
        mode: str,
        percentiles: tuple[float, float] = (1.0, 99.8),
        intensity_limits: tuple[float, float] | None = None,
    ):
        metadata = _layer_metadata(layer)
        mode = str(mode).strip().lower()
        if mode not in {"percentile", "intensity"}:
            mode = "percentile"
        metadata["reference_contrast_mode"] = mode

        if mode == "intensity":
            if intensity_limits is None:
                intensity_limits = tuple(float(v) for v in getattr(layer, "contrast_limits", finite_data_limits(np.asarray(layer.data))))
            low, high = float(intensity_limits[0]), float(intensity_limits[1])
            if high <= low:
                high = low + 1e-9
            layer.contrast_limits = (low, high)
            metadata["reference_contrast_limits"] = (low, high)
            return

        low_pct = float(percentiles[0])
        high_pct = float(percentiles[1])
        if high_pct <= low_pct:
            high_pct = min(100.0, low_pct + 0.1)
        layer.contrast_limits = auto_contrast_limits(np.asarray(layer.data), low_pct=low_pct, high_pct=high_pct)
        metadata["reference_contrast_percentiles"] = (low_pct, high_pct)
        metadata["reference_contrast_limits"] = tuple(float(v) for v in layer.contrast_limits)

    def _apply_reference_layer_gamma(layer, gamma: float):
        metadata = _layer_metadata(layer)
        gamma = float(gamma)
        if not np.isfinite(gamma) or gamma <= 0:
            gamma = 1.0
        metadata["reference_gamma"] = gamma
        try:
            layer.gamma = gamma
        except Exception:
            pass

    def _read_reference_display_settings_from_zarr(reference_key: str) -> dict[str, Any]:
        try:
            root = zarr.open_group(host_zarr_path, mode="r", use_consolidated=False)
            raw = root["images"][reference_key].attrs.get("if_display_settings", {})
        except Exception:
            raw = {}
        return dict(raw) if isinstance(raw, Mapping) else {}

    def _saved_reference_display_settings(image_attrs: Mapping[str, Any], channel_index: int, reference_key: str = "") -> dict[str, Any]:
        raw = _read_reference_display_settings_from_zarr(reference_key) if reference_key else {}
        if not isinstance(raw, Mapping) or not raw:
            raw = image_attrs.get("if_display_settings", {}) if isinstance(image_attrs, Mapping) else {}
        if not isinstance(raw, Mapping):
            return {}
        value = raw.get(str(channel_index), raw.get(channel_index, {}))
        return dict(value) if isinstance(value, Mapping) else {}

    def _apply_saved_reference_display_metadata(layer, saved: Mapping[str, Any]):
        if not saved:
            return
        metadata = _layer_metadata(layer)
        for key in (
            "display_name",
            "visible",
            "color_choice",
            "contrast_mode",
            "contrast_percentiles",
            "contrast_limits",
            "gamma",
        ):
            if key not in saved:
                continue
            value = saved[key]
            if key == "display_name" and str(value).strip():
                layer.name = str(value).strip()
            elif key == "visible":
                layer.visible = bool(value)
            elif key == "color_choice":
                metadata["reference_color_choice"] = str(value)
            elif key == "contrast_mode":
                metadata["reference_contrast_mode"] = str(value)
            elif key == "contrast_percentiles" and isinstance(value, (list, tuple)) and len(value) >= 2:
                metadata["reference_contrast_percentiles"] = (float(value[0]), float(value[1]))
            elif key == "contrast_limits" and isinstance(value, (list, tuple)) and len(value) >= 2:
                metadata["reference_contrast_limits"] = (float(value[0]), float(value[1]))
            elif key == "gamma":
                metadata["reference_gamma"] = float(value)

    def _refresh_if_toolbox_widgets(preferred_layer_name: str | None = None):
        try:
            rebuild_if_layer_controls()
        except NameError:
            pass

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
                _apply_saved_reference_display_metadata(layer, _saved_reference_display_settings(image_attrs, idx, key))
                metadata["reference_key"] = key
                metadata["reference_channel_index"] = idx
                metadata["reference_default_name"] = channel_names[idx]
                metadata["reference_default_rgb"] = default_rgb
                metadata.setdefault("reference_color_choice", "metadata")
                metadata.setdefault("reference_contrast_mode", "percentile")
                metadata.setdefault("reference_contrast_percentiles", (1.0, 99.8))
                metadata.setdefault("reference_contrast_limits", tuple(float(v) for v in getattr(layer, "contrast_limits", finite_data_limits(np.asarray(layer.data)))))
                metadata.setdefault("reference_gamma", float(getattr(layer, "gamma", 1.0)))
                layer.opacity = 1.0
                layer.blending = "translucent"
                _set_reference_layer_color(layer, str(metadata.get("reference_color_choice", "metadata")))
                _apply_reference_layer_contrast(
                    layer,
                    str(metadata.get("reference_contrast_mode", "percentile")),
                    percentiles=tuple(float(v) for v in metadata.get("reference_contrast_percentiles", (1.0, 99.8))),
                    intensity_limits=tuple(float(v) for v in metadata.get("reference_contrast_limits", getattr(layer, "contrast_limits", (0.0, 1.0)))),
                )
                _apply_reference_layer_gamma(layer, float(metadata.get("reference_gamma", 1.0)))
            reference_layers[key] = existing_layers
        else:
            layer = existing_layers[0] if existing_layers else None
            if layer is not None:
                layer.data = arr
                layer.visible = visible
            else:
                layer = viewer.add_image(arr, name=key, visible=visible)
            metadata = _layer_metadata(layer)
            _apply_saved_reference_display_metadata(layer, _saved_reference_display_settings(image_attrs, 0, key))
            metadata["reference_key"] = key
            metadata["reference_channel_index"] = 0
            metadata["reference_default_name"] = key
            metadata.setdefault("reference_color_choice", "metadata")
            metadata.setdefault("reference_contrast_mode", "percentile")
            metadata.setdefault("reference_contrast_percentiles", (1.0, 99.8))
            metadata.setdefault("reference_contrast_limits", tuple(float(v) for v in getattr(layer, "contrast_limits", finite_data_limits(np.asarray(layer.data)))))
            metadata.setdefault("reference_gamma", float(getattr(layer, "gamma", 1.0)))
            layer.opacity = 1.0
            layer.blending = "translucent"
            if metadata.get("reference_default_rgb") is None:
                metadata["reference_default_rgb"] = REFERENCE_CHANNEL_COLOR_PRESETS["white"]
            _set_reference_layer_color(layer, str(metadata.get("reference_color_choice", "metadata")))
            _apply_reference_layer_contrast(
                layer,
                str(metadata.get("reference_contrast_mode", "percentile")),
                percentiles=tuple(float(v) for v in metadata.get("reference_contrast_percentiles", (1.0, 99.8))),
                intensity_limits=tuple(float(v) for v in metadata.get("reference_contrast_limits", getattr(layer, "contrast_limits", (0.0, 1.0)))),
            )
            _apply_reference_layer_gamma(layer, float(metadata.get("reference_gamma", 1.0)))
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
    annotation_label_colors = {}
    annotation_palette = [
        "#ffcc00",
        "#00d1ff",
        "#ff5f87",
        "#7bff57",
        "#c18bff",
        "#ff9f1c",
        "#2ec4b6",
        "#e71d36",
    ]
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

    def get_annotation_color(label: str) -> str:
        cleaned = str(label).strip()
        if not cleaned:
            cleaned = "(unlabeled)"
        if cleaned not in annotation_label_colors:
            color_idx = len(annotation_label_colors) % len(annotation_palette)
            annotation_label_colors[cleaned] = annotation_palette[color_idx]
        return annotation_label_colors[cleaned]

    def annotation_edge_colors(labels: Iterable[str], fallback_label: str) -> list[str]:
        fallback = str(fallback_label).strip() or "(unlabeled)"
        return [get_annotation_color(str(label).strip() or fallback) for label in labels]

    def apply_annotation_visuals():
        for layer in annotation_shape_layers.values():
            try:
                metadata = _layer_metadata(layer)
                fallback_label = str(metadata.get("annotation_shape_key", layer.name))
                labels = []
                try:
                    labels = list(layer.properties.get("label", []))
                except Exception:
                    labels = []
                if labels:
                    layer.edge_color = annotation_edge_colors(labels, fallback_label)
                else:
                    layer.edge_color = get_annotation_color(fallback_label)
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

    def transformed_msi_xy(state: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        coreg_dataset = state["dataset"]
        empty_xy = np.empty((0, 2), dtype=float)
        empty_mask = np.zeros(len(coreg_dataset.x_coords), dtype=bool)
        transform_xy = np.asarray(state["current_transform_xy"], dtype=float)
        if transform_xy.shape != (3, 3) or not np.all(np.isfinite(transform_xy)):
            return empty_xy, empty_mask
        xy1 = np.column_stack(
            [coreg_dataset.x_coords.astype(float), coreg_dataset.y_coords.astype(float), np.ones_like(coreg_dataset.x_coords, dtype=float)]
        )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            xy_t = (transform_xy @ xy1.T).T[:, :2]
        finite = np.all(np.isfinite(xy_t), axis=1)
        return xy_t[finite], finite

    def compute_annotation_region_mask(
        state: dict[str, Any],
        roi_shape_key: str,
        region_label: str = "(all regions)",
    ) -> np.ndarray:
        coreg_dataset = state["dataset"]
        if roi_shape_key not in coreg_dataset.sdata.shapes:
            return np.zeros(len(coreg_dataset.x_coords), dtype=bool)
        rois = coreg_dataset.sdata.transform_element_to_coordinate_system(roi_shape_key, registered_cs)
        xy_t, finite = transformed_msi_xy(state)
        if not np.any(finite):
            return np.zeros(len(coreg_dataset.x_coords), dtype=bool)
        points = np.array([Point(px, py) for px, py in xy_t], dtype=object)
        inside = [
            np.fromiter((point.within(rois.geometry.iloc[idx]) for point in points), dtype=bool, count=len(points))
            for idx in range(len(rois))
        ]
        inside = np.vstack(inside) if inside else np.zeros((0, len(points)), dtype=bool)
        if region_label == "(all regions)" or "_annotation_label" not in rois.columns:
            selected_finite = np.any(inside, axis=0) if inside.size else np.zeros(len(points), dtype=bool)
            selected = np.zeros(len(coreg_dataset.x_coords), dtype=bool)
            selected[finite] = selected_finite
            return selected
        matching_idxs = [
            idx for idx, value in enumerate(rois["_annotation_label"])
            if str(value).strip() == str(region_label).strip()
        ]
        selected_finite = np.any(inside[matching_idxs], axis=0) if matching_idxs else np.zeros(len(points), dtype=bool)
        selected = np.zeros(len(coreg_dataset.x_coords), dtype=bool)
        selected[finite] = selected_finite
        return selected

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
        xy_t, finite = transformed_msi_xy(source_state)
        if not np.any(finite):
            return source_state, shape_key, np.zeros(len(coreg_dataset.x_coords), dtype=bool), default_label
        points = np.array([Point(px, py) for px, py in xy_t], dtype=object)
        inside = [
            np.fromiter((point.within(transformed_subset.geometry.iloc[idx]) for point in points), dtype=bool, count=len(points))
            for idx in range(len(transformed_subset))
        ]
        selected_finite = np.any(np.vstack(inside), axis=0) if inside else np.zeros(len(points), dtype=bool)
        selected_mask = np.zeros(len(coreg_dataset.x_coords), dtype=bool)
        selected_mask[finite] = selected_finite
        return source_state, shape_key, selected_mask, default_label

    def refresh_annotation_widget_choices():
        keys = current_annotation_shape_keys()
        choices = ["(none)"] + keys if keys else ["(none)"]
        if "remove_geojson_annotations" in locals():
            remove_geojson_annotations.annotation_key.choices = choices
            if remove_geojson_annotations.annotation_key.value not in choices:
                remove_geojson_annotations.annotation_key.value = choices[0]
        if "delete_threshold_annotation_widget" in locals():
            delete_threshold_annotation_widget.annotation_key.choices = choices
            if delete_threshold_annotation_widget.annotation_key.value not in choices:
                delete_threshold_annotation_widget.annotation_key.value = choices[0]
        if "rescale_geojson_annotations_widget" in locals():
            rescale_geojson_annotations_widget.annotation_key.choices = choices
            if rescale_geojson_annotations_widget.annotation_key.value not in choices:
                rescale_geojson_annotations_widget.annotation_key.value = choices[0]

    def add_annotation_shape_layers(state: dict[str, Any], shape_keys: Iterable[str] | None = None):
        source_dataset = state["dataset"]
        keys = [key for key in (shape_keys or source_dataset.sdata.shapes.keys()) if "pixels" not in key.lower()]
        for key in keys:
            if key not in source_dataset.sdata.shapes:
                try:
                    source_dataset.sdata = sd.read_zarr(source_dataset.zarr_path)
                except Exception:
                    pass
            if key not in source_dataset.sdata.shapes:
                continue
            try:
                gdf = source_dataset.sdata.transform_element_to_coordinate_system(key, registered_cs)
            except Exception:
                if key not in source_dataset.sdata.shapes:
                    continue
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
            edge_colors = annotation_edge_colors(shape_labels, key)
            if layer_name in annotation_shape_layers:
                annotation_shape_layers[layer_name].data = shape_data
                try:
                    annotation_shape_layers[layer_name].properties = {"label": np.asarray(shape_labels, dtype=object)}
                    annotation_shape_layers[layer_name].edge_color = edge_colors
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
                edge_color=edge_colors,
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
        apply_ion_contrast_to_active_layer(img)
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
        apply_ion_contrast_to_active_layer(img)
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
        contrast_mode={
            "widget_type": "ComboBox",
            "choices": ["percentile", "absolute"],
            "label": "Contrast scale",
        },
        contrast_percentiles={
            "widget_type": "FloatRangeSlider",
            "min": 0.0,
            "max": 100.0,
            "step": 0.1,
            "label": "Contrast percentiles",
        },
        absolute_low={
            "widget_type": "LineEdit",
            "label": "Absolute low",
        },
        absolute_high={
            "widget_type": "LineEdit",
            "label": "Absolute high",
        },
        auto_call=True,
    )
    def ion_display_options(
        normalize_to_tic=True,
        contrast_mode: str = "percentile",
        contrast_percentiles: tuple[float, float] = (1.0, 99.5),
        absolute_low: str = "0.0",
        absolute_high: str = "1.0",
    ):
        state = get_active_state()
        state["current_normalize_to_tic"] = bool(normalize_to_tic)
        mode = str(contrast_mode).lower()
        if mode not in {"percentile", "absolute"}:
            mode = "percentile"
        state["current_contrast_mode"] = mode
        low, high = contrast_percentiles
        low = float(low)
        high = float(high)
        if high <= low:
            high = min(100.0, low + 0.1)
            ion_display_options.contrast_percentiles.value = (low, high)
        state["current_contrast_low_pct"] = low
        state["current_contrast_high_pct"] = high
        try:
            absolute_low = float(str(absolute_low).strip())
            absolute_high = float(str(absolute_high).strip())
        except Exception:
            if mode == "absolute":
                return
            absolute_low = float(state.get("current_contrast_low", 0.0))
            absolute_high = float(state.get("current_contrast_high", 1.0))
        if absolute_high <= absolute_low:
            absolute_high = absolute_low + 1e-9
            ion_display_options.absolute_high.value = f"{absolute_high:g}"
        state["current_contrast_low"] = absolute_low
        state["current_contrast_high"] = absolute_high
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

    threshold_preview_updates_enabled = False
    suppress_threshold_absolute_update = False

    def update_threshold_preview_from_widget():
        nonlocal suppress_threshold_absolute_update
        source_state = get_active_state()
        source_dataset = source_state["dataset"]
        try:
            target_dataset_choice = str(threshold_map_to_dataset.value)
            target_dataset_key = dataset_key_from_choice(target_dataset_choice, str(source_state["id"]))
            target_state = datasets.get(str(target_dataset_key), source_state)
            target_dataset = target_state["dataset"]
            parsed_target_mz = float(str(threshold_target_mz.value).strip())
            ppm = float(threshold_ppm_tolerance.value)
            normalize = bool(threshold_normalize_to_tic.value)
            percentile = float(threshold_percentile.value)
            prefilter_shape_key = str(threshold_prefilter_annotation.value)
            prefilter_region_label = str(threshold_prefilter_region.value)
        except Exception as exc:
            QMessageBox.warning(None, "MSI Threshold Preview", str(exc))
            return
        for dataset_state in datasets.values():
            preview_layer = dataset_state.get("threshold_preview_layer")
            if preview_layer is not None and dataset_state is not target_state:
                preview_layer.visible = False
        if prefilter_shape_key not in target_dataset.sdata.shapes:
            prefilter_shape_key = "(none)"
            prefilter_region_label = "(all regions)"
        prefilter_region_choices = annotation_region_choices(target_dataset, prefilter_shape_key)
        threshold_prefilter_region.choices = prefilter_region_choices
        if prefilter_region_label not in prefilter_region_choices:
            prefilter_region_label = prefilter_region_choices[0]
            threshold_prefilter_region.value = prefilter_region_label
        target_state["threshold_preview_prefilter_shape_key"] = prefilter_shape_key
        target_state["threshold_preview_prefilter_region_label"] = prefilter_region_label
        cached_source_transform = target_state.get("threshold_preview_source_transform_xy")
        cached_target_transform = target_state.get("threshold_preview_target_transform_xy")
        needs_recompute = (
            target_state.get("threshold_preview_values") is None
            or target_state.get("threshold_preview_source_dataset_id") != str(source_state["id"])
            or cached_source_transform is None
            or cached_target_transform is None
            or not np.allclose(np.asarray(cached_source_transform, dtype=float), source_state["current_transform_xy"])
            or not np.allclose(np.asarray(cached_target_transform, dtype=float), target_state["current_transform_xy"])
            or target_state.get("threshold_preview_mz") != parsed_target_mz
            or target_state.get("threshold_preview_ppm") != ppm
            or target_state.get("threshold_preview_normalize_to_tic") != normalize
        )
        if needs_recompute:
            indices = source_dataset.find_feature_indices_from_mz(parsed_target_mz, ppm)
            if indices.size == 0:
                idx, _ = source_dataset.find_feature_idx_from_mz(parsed_target_mz, float("inf"))
                if idx is None:
                    QMessageBox.warning(None, "MSI Threshold Preview", f"No m/z feature found near {parsed_target_mz:g} in the active MSI dataset.")
                    return
                indices = np.array([idx], dtype=int)
            source_img = source_dataset.reconstruct_ion_image(indices, normalize_to_tic=normalize)
            if str(source_state["id"]) == str(target_state["id"]):
                values = source_img[target_dataset.y_coords, target_dataset.x_coords]
                target_state["threshold_preview_img"] = source_img
            else:
                values = _sample_msi_values_at_msi_pixels(
                    source_img,
                    source_dataset,
                    source_state["current_transform_xy"],
                    target_dataset,
                    target_state["current_transform_xy"],
                )
                target_state["threshold_preview_img"] = None
            target_state["threshold_preview_values"] = np.asarray(values, dtype=float)
            target_state["threshold_preview_feature_indices"] = np.asarray(indices, dtype=int)
            target_state["threshold_preview_source_dataset_id"] = str(source_state["id"])
            target_state["threshold_preview_source_transform_xy"] = np.asarray(source_state["current_transform_xy"], dtype=float).copy()
            target_state["threshold_preview_target_transform_xy"] = np.asarray(target_state["current_transform_xy"], dtype=float).copy()
            target_state["threshold_preview_mz"] = parsed_target_mz
            target_state["threshold_preview_ppm"] = ppm
            target_state["threshold_preview_normalize_to_tic"] = normalize
        values = np.asarray(target_state["threshold_preview_values"], dtype=float)
        allowed = np.ones(len(values), dtype=bool)
        if prefilter_shape_key != "(none)":
            allowed = compute_annotation_region_mask(target_state, prefilter_shape_key, prefilter_region_label)
            if not np.any(allowed):
                QMessageBox.warning(None, "MSI Threshold Preview", "No MSI pixels fall inside the selected prefilter annotation.")
                return
        finite_values = values[np.isfinite(values) & allowed]
        if finite_values.size == 0:
            QMessageBox.warning(None, "MSI Threshold Preview", "No finite MSI intensities available for this m/z.")
            return
        threshold = float(np.percentile(finite_values, percentile))
        threshold_value_label.setText(f"Threshold: {threshold:.6g}")
        preview = np.zeros((target_dataset.ny, target_dataset.nx), dtype=np.uint8)
        below = (values < threshold) & allowed
        above = (values >= threshold) & allowed
        preview[target_dataset.y_coords[below], target_dataset.x_coords[below]] = 1
        preview[target_dataset.y_coords[above], target_dataset.x_coords[above]] = 2
        preview_layer = ensure_threshold_preview_layer(target_state)
        preview_layer.data = preview
        preview_layer.name = f"{source_state['label']} mapped to {target_state['label']} threshold preview {parsed_target_mz:.4f}"
        preview_layer.visible = True
        threshold_selected_count_label.setText(f"Above: {int(np.count_nonzero(above))} / {int(np.count_nonzero(allowed))} pixels")
        try:
            suppress_threshold_absolute_update = True
            threshold_absolute_value.value = threshold
        except Exception:
            pass
        finally:
            suppress_threshold_absolute_update = False

    @magicgui(
        map_to_dataset={"widget_type": "ComboBox", "choices": current_dataset_choices(), "label": "Map to MSI dataset"},
        target_mz={"widget_type": "LineEdit"},
        ppm_tolerance={"widget_type": "FloatSpinBox", "min": 0.1, "step": 0.5},
        normalize_to_tic={"widget_type": "CheckBox", "text": "Normalize to TIC"},
        prefilter_annotation={"widget_type": "ComboBox", "choices": ["(none)"], "label": "Prefilter annotation"},
        prefilter_region={"widget_type": "ComboBox", "choices": ["(all regions)"], "label": "Prefilter region"},
        auto_call=False,
        call_button=False,
    )
    def threshold_preview_controls(
        map_to_dataset: str = dataset_choice_text(initial_state),
        target_mz: str = f"{float(initial_state['dataset'].mz_values[initial_state['current_feature_idx']]):.4f}",
        ppm_tolerance: float = 5.0,
        normalize_to_tic: bool = True,
        prefilter_annotation: str = "(none)",
        prefilter_region: str = "(all regions)",
    ):
        schedule_threshold_preview_update()

    threshold_map_to_dataset = threshold_preview_controls.map_to_dataset
    threshold_target_mz = threshold_preview_controls.target_mz
    threshold_ppm_tolerance = threshold_preview_controls.ppm_tolerance
    threshold_normalize_to_tic = threshold_preview_controls.normalize_to_tic
    threshold_prefilter_annotation = threshold_preview_controls.prefilter_annotation
    threshold_prefilter_region = threshold_preview_controls.prefilter_region

    @magicgui(
        percentile={
            "widget_type": "FloatSlider",
            "min": 0.0,
            "max": 100.0,
            "step": 0.1,
            "label": "Threshold percentile",
        },
        auto_call=False,
    )
    def threshold_percentile_controls(percentile: float = 90.0):
        schedule_threshold_preview_update()

    threshold_percentile = threshold_percentile_controls.percentile
    threshold_value_label = QLabel("Threshold: --")
    threshold_selected_count_label = QLabel("Above: --")

    threshold_preview_update_timer = QTimer()
    threshold_preview_update_timer.setSingleShot(True)
    threshold_preview_update_timer.setInterval(350)

    def run_scheduled_threshold_preview_update():
        if not threshold_preview_updates_enabled:
            return
        _run_with_busy_dialog(
            "MSI Threshold Preview",
            "Updating MSI threshold preview...",
            update_threshold_preview_from_widget,
        )

    def schedule_threshold_preview_update(*_args):
        if not threshold_preview_updates_enabled:
            return
        threshold_preview_update_timer.start()

    threshold_preview_update_timer.timeout.connect(run_scheduled_threshold_preview_update)

    @magicgui(
        threshold={"widget_type": "FloatSpinBox", "min": -1e15, "max": 1e15, "step": 0.001},
        annotation_name={"widget_type": "LineEdit"},
        annotation_label={"widget_type": "LineEdit", "label": "Label prefix"},
        call_button="Create Lower/Higher Annotation",
    )
    def create_msi_threshold_annotation_from_preview(
        threshold: float = 0.0,
        annotation_name: str = "",
        annotation_label: str = "",
    ):
        source_state = get_active_state()
        def create_annotation():
            target_dataset_key = dataset_key_from_choice(str(threshold_map_to_dataset.value), str(source_state["id"]))
            target_state = datasets.get(str(target_dataset_key), source_state)
            parsed_target_mz = float(str(threshold_target_mz.value).strip())
            prefilter_shape_key = str(threshold_prefilter_annotation.value)
            prefilter_region_label = str(threshold_prefilter_region.value)
            prefilter_mask = None
            if prefilter_shape_key in target_state["dataset"].sdata.shapes:
                prefilter_mask = compute_annotation_region_mask(target_state, prefilter_shape_key, prefilter_region_label)
            if str(source_state["id"]) != str(target_state["id"]):
                return create_pooled_msi_threshold_annotation(
                    target_state["dataset"].zarr_path,
                    source_table_key=source_state["dataset"].table_key,
                    source_tic_key=source_state["dataset"].tic_key,
                    target_table_key=target_state["dataset"].table_key,
                    target_tic_key=target_state["dataset"].tic_key,
                    target_mz=parsed_target_mz,
                    ppm_tolerance=float(threshold_ppm_tolerance.value),
                    threshold=float(threshold),
                    normalize_to_tic=bool(threshold_normalize_to_tic.value),
                    source_transform_xy=source_state["current_transform_xy"],
                    target_transform_xy=target_state["current_transform_xy"],
                    prefilter_mask=prefilter_mask,
                    prefilter_shape_key=prefilter_shape_key if prefilter_mask is not None else "",
                    prefilter_region_label=prefilter_region_label if prefilter_mask is not None else "",
                    annotation_name=str(annotation_name),
                    annotation_label=str(annotation_label),
                    registered_cs=registered_cs,
                )
            return create_msi_threshold_annotation(
                target_state["dataset"].zarr_path,
                table_key=target_state["dataset"].table_key,
                tic_key=target_state["dataset"].tic_key,
                target_mz=parsed_target_mz,
                ppm_tolerance=float(threshold_ppm_tolerance.value),
                threshold=float(threshold),
                normalize_to_tic=bool(threshold_normalize_to_tic.value),
                transform_xy=target_state["current_transform_xy"],
                prefilter_mask=prefilter_mask,
                prefilter_shape_key=prefilter_shape_key if prefilter_mask is not None else "",
                prefilter_region_label=prefilter_region_label if prefilter_mask is not None else "",
                annotation_name=str(annotation_name),
                annotation_label=str(annotation_label),
                registered_cs=registered_cs,
            )

        try:
            key = _run_with_busy_dialog(
                "Create MSI Threshold Annotation",
                "Creating MSI threshold annotation...\nThis can take a little while for large datasets.",
                create_annotation,
            )
        except Exception as exc:
            QMessageBox.warning(None, "Create MSI Threshold Annotation", str(exc))
            return
        def refresh_annotation_layers():
            for dataset_state in datasets.values():
                dataset_state["dataset"].sdata = sd.read_zarr(dataset_state["dataset"].zarr_path)
                add_annotation_shape_layers(dataset_state, [key])
        _run_with_busy_dialog(
            "Create MSI Threshold Annotation",
            "Refreshing annotation layers...",
            refresh_annotation_layers,
        )
        sync_controls_to_active_dataset()
        try:
            roi_mask_controls.roi_shape_key.value = key
        except Exception:
            pass

    threshold_absolute_value = create_msi_threshold_annotation_from_preview.threshold
    try:
        threshold_absolute_value.native.setDecimals(12)
        threshold_absolute_value.native.setSingleStep(1e-6)
    except Exception:
        pass

    def update_threshold_preview_from_absolute_widget(*_args):
        if suppress_threshold_absolute_update:
            return
        source_state = get_active_state()
        source_dataset = source_state["dataset"]
        try:
            target_dataset_choice = str(threshold_map_to_dataset.value)
            target_dataset_key = dataset_key_from_choice(target_dataset_choice, str(source_state["id"]))
            target_state = datasets.get(str(target_dataset_key), source_state)
            target_dataset = target_state["dataset"]
            parsed_target_mz = float(str(threshold_target_mz.value).strip())
            ppm = float(threshold_ppm_tolerance.value)
            normalize = bool(threshold_normalize_to_tic.value)
            threshold = float(threshold_absolute_value.value)
            prefilter_shape_key = str(threshold_prefilter_annotation.value)
            prefilter_region_label = str(threshold_prefilter_region.value)
        except Exception as exc:
            QMessageBox.warning(None, "MSI Threshold Preview", str(exc))
            return
        for dataset_state in datasets.values():
            preview_layer = dataset_state.get("threshold_preview_layer")
            if preview_layer is not None and dataset_state is not target_state:
                preview_layer.visible = False
        if prefilter_shape_key not in target_dataset.sdata.shapes:
            prefilter_shape_key = "(none)"
            prefilter_region_label = "(all regions)"
        prefilter_region_choices = annotation_region_choices(target_dataset, prefilter_shape_key)
        threshold_prefilter_region.choices = prefilter_region_choices
        if prefilter_region_label not in prefilter_region_choices:
            prefilter_region_label = prefilter_region_choices[0]
            threshold_prefilter_region.value = prefilter_region_label
        target_state["threshold_preview_prefilter_shape_key"] = prefilter_shape_key
        target_state["threshold_preview_prefilter_region_label"] = prefilter_region_label
        cached_source_transform = target_state.get("threshold_preview_source_transform_xy")
        cached_target_transform = target_state.get("threshold_preview_target_transform_xy")
        needs_recompute = (
            target_state.get("threshold_preview_values") is None
            or target_state.get("threshold_preview_source_dataset_id") != str(source_state["id"])
            or cached_source_transform is None
            or cached_target_transform is None
            or not np.allclose(np.asarray(cached_source_transform, dtype=float), source_state["current_transform_xy"])
            or not np.allclose(np.asarray(cached_target_transform, dtype=float), target_state["current_transform_xy"])
            or target_state.get("threshold_preview_mz") != parsed_target_mz
            or target_state.get("threshold_preview_ppm") != ppm
            or target_state.get("threshold_preview_normalize_to_tic") != normalize
        )
        if needs_recompute:
            indices = source_dataset.find_feature_indices_from_mz(parsed_target_mz, ppm)
            if indices.size == 0:
                idx, _ = source_dataset.find_feature_idx_from_mz(parsed_target_mz, float("inf"))
                if idx is None:
                    QMessageBox.warning(None, "MSI Threshold Preview", f"No m/z feature found near {parsed_target_mz:g} in the active MSI dataset.")
                    return
                indices = np.array([idx], dtype=int)
            source_img = source_dataset.reconstruct_ion_image(indices, normalize_to_tic=normalize)
            if str(source_state["id"]) == str(target_state["id"]):
                values = source_img[target_dataset.y_coords, target_dataset.x_coords]
                target_state["threshold_preview_img"] = source_img
            else:
                values = _sample_msi_values_at_msi_pixels(
                    source_img,
                    source_dataset,
                    source_state["current_transform_xy"],
                    target_dataset,
                    target_state["current_transform_xy"],
                )
                target_state["threshold_preview_img"] = None
            target_state["threshold_preview_values"] = np.asarray(values, dtype=float)
            target_state["threshold_preview_feature_indices"] = np.asarray(indices, dtype=int)
            target_state["threshold_preview_source_dataset_id"] = str(source_state["id"])
            target_state["threshold_preview_source_transform_xy"] = np.asarray(source_state["current_transform_xy"], dtype=float).copy()
            target_state["threshold_preview_target_transform_xy"] = np.asarray(target_state["current_transform_xy"], dtype=float).copy()
            target_state["threshold_preview_mz"] = parsed_target_mz
            target_state["threshold_preview_ppm"] = ppm
            target_state["threshold_preview_normalize_to_tic"] = normalize
        values = np.asarray(target_state["threshold_preview_values"], dtype=float)
        allowed = np.ones(len(values), dtype=bool)
        if prefilter_shape_key != "(none)":
            allowed = compute_annotation_region_mask(target_state, prefilter_shape_key, prefilter_region_label)
            if not np.any(allowed):
                QMessageBox.warning(None, "MSI Threshold Preview", "No MSI pixels fall inside the selected prefilter annotation.")
                return
        finite_values = values[np.isfinite(values) & allowed]
        if finite_values.size == 0:
            QMessageBox.warning(None, "MSI Threshold Preview", "No finite MSI intensities available for this m/z.")
            return
        threshold_value_label.setText(f"Threshold: {threshold:.6g}")
        preview = np.zeros((target_dataset.ny, target_dataset.nx), dtype=np.uint8)
        below = (values < threshold) & allowed
        above = (values >= threshold) & allowed
        preview[target_dataset.y_coords[below], target_dataset.x_coords[below]] = 1
        preview[target_dataset.y_coords[above], target_dataset.x_coords[above]] = 2
        preview_layer = ensure_threshold_preview_layer(target_state)
        preview_layer.data = preview
        preview_layer.name = f"{source_state['label']} mapped to {target_state['label']} threshold preview {parsed_target_mz:.4f}"
        preview_layer.visible = True
        threshold_selected_count_label.setText(f"Above: {int(np.count_nonzero(above))} / {int(np.count_nonzero(allowed))} pixels")

    try:
        threshold_absolute_value.changed.connect(update_threshold_preview_from_absolute_widget)
    except Exception:
        pass
    for widget in (
        threshold_map_to_dataset,
        threshold_target_mz,
        threshold_ppm_tolerance,
        threshold_normalize_to_tic,
        threshold_prefilter_annotation,
        threshold_prefilter_region,
        threshold_percentile,
    ):
        try:
            widget.changed.connect(schedule_threshold_preview_update)
        except Exception:
            pass

    def _reference_intensity_from_layer(layer) -> np.ndarray:
        arr = np.asarray(layer.data)
        if arr.ndim == 2:
            return np.asarray(arr, dtype=float)
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            rgb = np.asarray(arr[..., :3], dtype=float)
            return np.dot(rgb, np.array([0.2126, 0.7152, 0.0722], dtype=float))
        if arr.ndim == 3 and arr.shape[0] in (3, 4):
            rgb = np.moveaxis(np.asarray(arr[:3], dtype=float), 0, -1)
            return np.dot(rgb, np.array([0.2126, 0.7152, 0.0722], dtype=float))
        raise ValueError(f"Reference layer {layer.name!r} is not a 2D fluorescence/intensity channel.")

    if_threshold_preview_updates_enabled = False
    suppress_if_threshold_absolute_update = False

    def update_if_threshold_preview_from_widget():
        nonlocal suppress_if_threshold_absolute_update
        state = get_active_state()
        coreg_dataset = state["dataset"]
        try:
            layer_name = str(if_threshold_reference_channel.value)
            percentile = float(if_threshold_percentile.value)
            prefilter_shape_key = str(if_threshold_prefilter_annotation.value)
            prefilter_region_label = str(if_threshold_prefilter_region.value)
        except Exception as exc:
            QMessageBox.warning(None, "IF Threshold Preview", str(exc))
            return
        layer = _get_reference_layer_by_name(layer_name)
        if layer is None:
            QMessageBox.warning(None, "IF Threshold Preview", "Select a fluorescence/reference channel first.")
            return
        if prefilter_shape_key not in coreg_dataset.sdata.shapes:
            prefilter_shape_key = "(none)"
            prefilter_region_label = "(all regions)"
        prefilter_region_choices = annotation_region_choices(coreg_dataset, prefilter_shape_key)
        if_threshold_prefilter_region.choices = prefilter_region_choices
        if prefilter_region_label not in prefilter_region_choices:
            prefilter_region_label = prefilter_region_choices[0]
            if_threshold_prefilter_region.value = prefilter_region_label

        try:
            reference_img = _reference_intensity_from_layer(layer)
            values = _sample_reference_values_at_msi_pixels(
                reference_img,
                coreg_dataset,
                state["current_transform_xy"],
            )
        except Exception as exc:
            QMessageBox.warning(None, "IF Threshold Preview", str(exc))
            return

        allowed = np.isfinite(values)
        if prefilter_shape_key != "(none)":
            allowed &= compute_annotation_region_mask(state, prefilter_shape_key, prefilter_region_label)
            if not np.any(allowed):
                QMessageBox.warning(None, "IF Threshold Preview", "No MSI pixels fall inside the selected prefilter annotation.")
                return
        finite_values = values[allowed]
        if finite_values.size == 0:
            QMessageBox.warning(None, "IF Threshold Preview", "No finite fluorescence intensities overlap the MSI pixels.")
            return
        threshold = float(np.percentile(finite_values, percentile))
        if_threshold_value_label.setText(f"Threshold: {threshold:.6g}")
        preview = np.zeros((coreg_dataset.ny, coreg_dataset.nx), dtype=np.uint8)
        below = (values < threshold) & allowed
        above = (values >= threshold) & allowed
        preview[coreg_dataset.y_coords[below], coreg_dataset.x_coords[below]] = 1
        preview[coreg_dataset.y_coords[above], coreg_dataset.x_coords[above]] = 2
        preview_layer = ensure_threshold_preview_layer(state)
        preview_layer.data = preview
        preview_layer.name = f"{state['label']} IF threshold preview"
        preview_layer.visible = True
        if_threshold_selected_count_label.setText(f"Above: {int(np.count_nonzero(above))} / {int(np.count_nonzero(allowed))} pixels")
        try:
            suppress_if_threshold_absolute_update = True
            if_threshold_absolute_value.value = threshold
        except Exception:
            pass
        finally:
            suppress_if_threshold_absolute_update = False

    @magicgui(
        reference_channel={"widget_type": "ComboBox", "choices": ["(none)"]},
        prefilter_annotation={"widget_type": "ComboBox", "choices": ["(none)"], "label": "Prefilter annotation"},
        prefilter_region={"widget_type": "ComboBox", "choices": ["(all regions)"], "label": "Prefilter region"},
        auto_call=False,
        call_button=False,
    )
    def if_threshold_preview_controls(
        reference_channel: str = "(none)",
        prefilter_annotation: str = "(none)",
        prefilter_region: str = "(all regions)",
    ):
        schedule_if_threshold_preview_update()

    if_threshold_reference_channel = if_threshold_preview_controls.reference_channel
    if_threshold_prefilter_annotation = if_threshold_preview_controls.prefilter_annotation
    if_threshold_prefilter_region = if_threshold_preview_controls.prefilter_region

    @magicgui(
        percentile={
            "widget_type": "FloatSlider",
            "min": 0.0,
            "max": 100.0,
            "step": 0.1,
            "label": "Threshold percentile",
        },
        auto_call=False,
    )
    def if_threshold_percentile_controls(percentile: float = 90.0):
        schedule_if_threshold_preview_update()

    if_threshold_percentile = if_threshold_percentile_controls.percentile
    if_threshold_value_label = QLabel("Threshold: --")
    if_threshold_selected_count_label = QLabel("Above: --")

    if_threshold_preview_update_timer = QTimer()
    if_threshold_preview_update_timer.setSingleShot(True)
    if_threshold_preview_update_timer.setInterval(350)

    def run_scheduled_if_threshold_preview_update():
        if not if_threshold_preview_updates_enabled:
            return
        _run_with_busy_dialog(
            "IF Threshold Preview",
            "Updating fluorescence threshold preview...",
            update_if_threshold_preview_from_widget,
        )

    def schedule_if_threshold_preview_update(*_args):
        if not if_threshold_preview_updates_enabled:
            return
        if_threshold_preview_update_timer.start()

    if_threshold_preview_update_timer.timeout.connect(run_scheduled_if_threshold_preview_update)

    @magicgui(
        threshold={"widget_type": "FloatSpinBox", "min": -1e15, "max": 1e15, "step": 0.001},
        annotation_name={"widget_type": "LineEdit"},
        annotation_label={"widget_type": "LineEdit", "label": "Label prefix"},
        call_button="Create Lower/Higher Annotation",
    )
    def create_if_threshold_annotation_from_preview(
        threshold: float = 0.0,
        annotation_name: str = "",
        annotation_label: str = "",
    ):
        state = get_active_state()
        layer = _get_reference_layer_by_name(str(if_threshold_reference_channel.value))
        if layer is None:
            QMessageBox.warning(None, "Create IF Threshold Annotation", "Select a fluorescence/reference channel first.")
            return
        metadata = _layer_metadata(layer)
        reference_key = str(metadata.get("reference_key", ""))
        channel_index = int(metadata.get("reference_channel_index", 0))
        if not reference_key:
            QMessageBox.warning(None, "Create IF Threshold Annotation", "Selected layer is missing reference image metadata.")
            return

        def create_annotation():
            prefilter_shape_key = str(if_threshold_prefilter_annotation.value)
            prefilter_region_label = str(if_threshold_prefilter_region.value)
            prefilter_mask = None
            if prefilter_shape_key in state["dataset"].sdata.shapes:
                prefilter_mask = compute_annotation_region_mask(state, prefilter_shape_key, prefilter_region_label)
            return create_reference_threshold_annotation(
                state["dataset"].zarr_path,
                table_key=state["dataset"].table_key,
                tic_key=state["dataset"].tic_key,
                reference_key=reference_key,
                channel_index=channel_index,
                channel_name=str(layer.name),
                threshold=float(threshold),
                transform_xy=state["current_transform_xy"],
                prefilter_mask=prefilter_mask,
                prefilter_shape_key=prefilter_shape_key if prefilter_mask is not None else "",
                prefilter_region_label=prefilter_region_label if prefilter_mask is not None else "",
                annotation_name=str(annotation_name),
                annotation_label=str(annotation_label),
                registered_cs=registered_cs,
            )

        try:
            key = _run_with_busy_dialog(
                "Create IF Threshold Annotation",
                "Creating fluorescence threshold annotation...\nSampling and writing masks can take a little while.",
                create_annotation,
            )
        except Exception as exc:
            QMessageBox.warning(None, "Create IF Threshold Annotation", str(exc))
            return
        def refresh_annotation_layers():
            for dataset_state in datasets.values():
                dataset_state["dataset"].sdata = sd.read_zarr(dataset_state["dataset"].zarr_path)
                add_annotation_shape_layers(dataset_state, [key])
        _run_with_busy_dialog(
            "Create IF Threshold Annotation",
            "Refreshing annotation layers...",
            refresh_annotation_layers,
        )
        sync_controls_to_active_dataset()
        try:
            roi_mask_controls.roi_shape_key.value = key
        except Exception:
            pass

    if_threshold_absolute_value = create_if_threshold_annotation_from_preview.threshold
    try:
        if_threshold_absolute_value.native.setDecimals(12)
        if_threshold_absolute_value.native.setSingleStep(1e-6)
    except Exception:
        pass

    def update_if_threshold_preview_from_absolute_widget(*_args):
        if suppress_if_threshold_absolute_update:
            return
        state = get_active_state()
        coreg_dataset = state["dataset"]
        try:
            layer_name = str(if_threshold_reference_channel.value)
            threshold = float(if_threshold_absolute_value.value)
            prefilter_shape_key = str(if_threshold_prefilter_annotation.value)
            prefilter_region_label = str(if_threshold_prefilter_region.value)
        except Exception as exc:
            QMessageBox.warning(None, "IF Threshold Preview", str(exc))
            return
        layer = _get_reference_layer_by_name(layer_name)
        if layer is None:
            QMessageBox.warning(None, "IF Threshold Preview", "Select a fluorescence/reference channel first.")
            return
        if prefilter_shape_key not in coreg_dataset.sdata.shapes:
            prefilter_shape_key = "(none)"
            prefilter_region_label = "(all regions)"
        prefilter_region_choices = annotation_region_choices(coreg_dataset, prefilter_shape_key)
        if_threshold_prefilter_region.choices = prefilter_region_choices
        if prefilter_region_label not in prefilter_region_choices:
            prefilter_region_label = prefilter_region_choices[0]
            if_threshold_prefilter_region.value = prefilter_region_label

        try:
            reference_img = _reference_intensity_from_layer(layer)
            values = _sample_reference_values_at_msi_pixels(
                reference_img,
                coreg_dataset,
                state["current_transform_xy"],
            )
        except Exception as exc:
            QMessageBox.warning(None, "IF Threshold Preview", str(exc))
            return

        allowed = np.isfinite(values)
        if prefilter_shape_key != "(none)":
            allowed &= compute_annotation_region_mask(state, prefilter_shape_key, prefilter_region_label)
            if not np.any(allowed):
                QMessageBox.warning(None, "IF Threshold Preview", "No MSI pixels fall inside the selected prefilter annotation.")
                return
        finite_values = values[allowed]
        if finite_values.size == 0:
            QMessageBox.warning(None, "IF Threshold Preview", "No finite fluorescence intensities overlap the MSI pixels.")
            return
        if_threshold_value_label.setText(f"Threshold: {threshold:.6g}")
        preview = np.zeros((coreg_dataset.ny, coreg_dataset.nx), dtype=np.uint8)
        below = (values < threshold) & allowed
        above = (values >= threshold) & allowed
        preview[coreg_dataset.y_coords[below], coreg_dataset.x_coords[below]] = 1
        preview[coreg_dataset.y_coords[above], coreg_dataset.x_coords[above]] = 2
        preview_layer = ensure_threshold_preview_layer(state)
        preview_layer.data = preview
        preview_layer.name = f"{state['label']} IF threshold preview"
        preview_layer.visible = True
        if_threshold_selected_count_label.setText(f"Above: {int(np.count_nonzero(above))} / {int(np.count_nonzero(allowed))} pixels")

    try:
        if_threshold_absolute_value.changed.connect(update_if_threshold_preview_from_absolute_widget)
    except Exception:
        pass
    for widget in (
        if_threshold_prefilter_annotation,
        if_threshold_prefilter_region,
        if_threshold_percentile,
    ):
        try:
            widget.changed.connect(schedule_if_threshold_preview_update)
        except Exception:
            pass

    def update_if_threshold_channel_defaults(*_args):
        state = get_active_state()
        selected_layer = str(if_threshold_reference_channel.value)
        refresh_if_threshold_choices(state, preferred_layer=selected_layer)
        schedule_if_threshold_preview_update()

    try:
        if_threshold_reference_channel.changed.connect(update_if_threshold_channel_defaults)
    except Exception:
        pass

    def refresh_threshold_prefilter_choices(
        state: dict[str, Any] | None = None,
        *,
        preferred_annotation: str | None = None,
        preferred_region: str | None = None,
    ):
        state = state or get_active_state()
        coreg_dataset = state["dataset"]
        shape_keys = [key for key in coreg_dataset.sdata.shapes.keys() if "pixels" not in key.lower()]
        annotation_choices = ["(none)"] + shape_keys
        threshold_prefilter_annotation.choices = annotation_choices
        current_annotation = str(
            preferred_annotation
            or state.get("threshold_preview_prefilter_shape_key")
            or threshold_prefilter_annotation.value
            or "(none)"
        )
        if current_annotation not in annotation_choices:
            current_annotation = "(none)"
        threshold_prefilter_annotation.value = current_annotation
        region_choices = annotation_region_choices(coreg_dataset, current_annotation)
        threshold_prefilter_region.choices = region_choices
        current_region = str(
            preferred_region
            or state.get("threshold_preview_prefilter_region_label")
            or threshold_prefilter_region.value
            or "(all regions)"
        )
        if current_region not in region_choices:
            current_region = region_choices[0]
        threshold_prefilter_region.value = current_region

    def update_threshold_prefilter_region_choices(*_args):
        state = get_active_state()
        selected_annotation = str(threshold_prefilter_annotation.value)
        refresh_threshold_prefilter_choices(state, preferred_annotation=selected_annotation)

    try:
        threshold_prefilter_annotation.changed.connect(update_threshold_prefilter_region_choices)
    except Exception:
        pass

    def refresh_if_threshold_choices(
        state: dict[str, Any] | None = None,
        *,
        preferred_layer: str | None = None,
        preferred_annotation: str | None = None,
        preferred_region: str | None = None,
    ):
        nonlocal suppress_if_threshold_absolute_update
        state = state or get_active_state()
        coreg_dataset = state["dataset"]
        channel_choices = _reference_channel_choice_names()
        if_threshold_reference_channel.choices = channel_choices
        current_layer = str(preferred_layer or if_threshold_reference_channel.value or "(none)")
        if current_layer not in channel_choices:
            current_layer = channel_choices[0]
        if_threshold_reference_channel.value = current_layer
        layer = _get_reference_layer_by_name(current_layer)
        layer_metadata = _layer_metadata(layer) if layer is not None else {}

        shape_keys = [key for key in coreg_dataset.sdata.shapes.keys() if "pixels" not in key.lower()]
        annotation_choices = ["(none)"] + shape_keys
        if_threshold_prefilter_annotation.choices = annotation_choices
        current_annotation = str(
            preferred_annotation
            or layer_metadata.get("if_threshold_prefilter_annotation")
            or if_threshold_prefilter_annotation.value
            or "(none)"
        )
        if current_annotation not in annotation_choices:
            current_annotation = "(none)"
        if_threshold_prefilter_annotation.value = current_annotation
        region_choices = annotation_region_choices(coreg_dataset, current_annotation)
        if_threshold_prefilter_region.choices = region_choices
        current_region = str(
            preferred_region
            or layer_metadata.get("if_threshold_prefilter_region")
            or if_threshold_prefilter_region.value
            or "(all regions)"
        )
        if current_region not in region_choices:
            current_region = region_choices[0]
        if_threshold_prefilter_region.value = current_region
        if "if_threshold_percentile" in layer_metadata:
            try:
                if_threshold_percentile.value = float(layer_metadata["if_threshold_percentile"])
            except Exception:
                pass
        if "if_threshold_value" in layer_metadata:
            try:
                suppress_if_threshold_absolute_update = True
                if_threshold_absolute_value.value = float(layer_metadata["if_threshold_value"])
            except Exception:
                pass
            finally:
                suppress_if_threshold_absolute_update = False

    def update_if_threshold_prefilter_region_choices(*_args):
        state = get_active_state()
        selected_annotation = str(if_threshold_prefilter_annotation.value)
        refresh_if_threshold_choices(state, preferred_annotation=selected_annotation)

    try:
        if_threshold_prefilter_annotation.changed.connect(update_if_threshold_prefilter_region_choices)
    except Exception:
        pass

    @magicgui(
        annotation_key={"widget_type": "ComboBox", "choices": ["(none)"]},
        call_button="Delete Annotation",
    )
    def delete_threshold_annotation_widget(annotation_key: str = "(none)"):
        if annotation_key == "(none)":
            return
        state = get_active_state()
        deleted = delete_geojson_annotations(state["dataset"].zarr_path, [annotation_key])
        if not deleted:
            QMessageBox.warning(None, "Delete Annotation", f"Could not find annotation {annotation_key!r} in the zarr.")
            return
        for dataset_state in datasets.values():
            dataset_state["dataset"].sdata = sd.read_zarr(dataset_state["dataset"].zarr_path)
        remove_annotation_shape_layers(deleted)
        sync_controls_to_active_dataset()

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

    def remove_optimization_preview(state: dict[str, Any], *, restore_visibility: bool = True):
        preview_layer = state.get("optimization_preview_layer")
        if preview_layer is not None:
            try:
                viewer.layers.remove(preview_layer)
            except Exception:
                pass
        if restore_visibility and state.get("optimization_previous_ion_visible") is not None:
            try:
                state["ion_layer"].visible = bool(state["optimization_previous_ion_visible"])
            except Exception:
                pass
        state["optimization_preview_layer"] = None
        state["optimization_candidate_transform_xy"] = None
        state["optimization_previous_ion_visible"] = None

    def resample_moving_on_fixed_crop(
        moving_img: np.ndarray,
        fixed_crop_sitk,
        crop_to_full_xy: np.ndarray,
        moving_to_fixed_xy: np.ndarray,
    ) -> np.ndarray:
        import SimpleITK as sitk

        moving_sitk = sitk.GetImageFromArray(normalize_image_for_registration(moving_img))
        fixed_to_moving_xy = np.linalg.inv(np.asarray(moving_to_fixed_xy, dtype=float))
        crop_fixed_to_moving_xy = fixed_to_moving_xy @ np.asarray(crop_to_full_xy, dtype=float)
        transform = sitk_affine_from_fixed_to_moving_matrix(crop_fixed_to_moving_xy)
        resampled = sitk.Resample(
            moving_sitk,
            fixed_crop_sitk,
            transform,
            sitk.sitkLinear,
            0.0,
            moving_sitk.GetPixelID(),
        )
        return sitk.GetArrayFromImage(resampled)

    def active_msi_registration_image(state: dict[str, Any]) -> np.ndarray:
        coreg_dataset = state["dataset"]
        return coreg_dataset.reconstruct_ion_image(
            state["current_feature_indices"],
            normalize_to_tic=bool(state["current_normalize_to_tic"]),
        )

    def fixed_mask_from_moving_footprint(
        fixed_shape: tuple[int, int],
        moving_shape: tuple[int, int],
        moving_to_fixed_xy: np.ndarray,
    ) -> np.ndarray:
        fixed_h, fixed_w = int(fixed_shape[0]), int(fixed_shape[1])
        moving_h, moving_w = int(moving_shape[0]), int(moving_shape[1])
        transform = np.asarray(moving_to_fixed_xy, dtype=float)
        corners = np.array(
            [
                [-0.5, -0.5, 1.0],
                [moving_w - 0.5, -0.5, 1.0],
                [moving_w - 0.5, moving_h - 0.5, 1.0],
                [-0.5, moving_h - 0.5, 1.0],
            ],
            dtype=float,
        )
        poly_xy = (transform @ corners.T).T[:, :2]
        min_x = max(0, int(np.floor(np.min(poly_xy[:, 0]))))
        max_x = min(fixed_w - 1, int(np.ceil(np.max(poly_xy[:, 0]))))
        min_y = max(0, int(np.floor(np.min(poly_xy[:, 1]))))
        max_y = min(fixed_h - 1, int(np.ceil(np.max(poly_xy[:, 1]))))
        mask = np.zeros((fixed_h, fixed_w), dtype=np.uint8)
        if min_x > max_x or min_y > max_y:
            return mask
        yy, xx = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
        sample_points = np.column_stack([xx.ravel().astype(float), yy.ravel().astype(float)])
        inside = MplPath(poly_xy).contains_points(sample_points, radius=1e-9)
        if np.any(inside):
            mask[yy.ravel()[inside], xx.ravel()[inside]] = 1
        return mask

    def fixed_mask_valid_for_transform(
        fixed_shape: tuple[int, int],
        moving_shape: tuple[int, int],
        fixed_to_moving_xy: np.ndarray,
    ) -> np.ndarray:
        fixed_h, fixed_w = int(fixed_shape[0]), int(fixed_shape[1])
        moving_h, moving_w = int(moving_shape[0]), int(moving_shape[1])
        try:
            moving_to_fixed_xy = np.linalg.inv(np.asarray(fixed_to_moving_xy, dtype=float))
        except Exception:
            return np.zeros((fixed_h, fixed_w), dtype=np.uint8)
        footprint = fixed_mask_from_moving_footprint(fixed_shape, moving_shape, moving_to_fixed_xy)
        yy, xx = np.nonzero(footprint)
        if yy.size == 0:
            return footprint
        fixed_points = np.column_stack([xx.astype(float), yy.astype(float), np.ones_like(xx, dtype=float)])
        moving_xy = (np.asarray(fixed_to_moving_xy, dtype=float) @ fixed_points.T).T[:, :2]
        inside = (
            (moving_xy[:, 0] >= -0.5)
            & (moving_xy[:, 1] >= -0.5)
            & (moving_xy[:, 0] <= moving_w - 0.5)
            & (moving_xy[:, 1] <= moving_h - 0.5)
        )
        mask = np.zeros((fixed_h, fixed_w), dtype=np.uint8)
        if np.any(inside):
            mask[yy[inside], xx[inside]] = 1
        return mask

    def sitk_overlap_count_for_transform(transform, fixed_mask_arr: np.ndarray, moving_shape: tuple[int, int], *, max_points: int = 20000) -> int:
        yy, xx = np.nonzero(fixed_mask_arr)
        if yy.size == 0:
            return 0
        if yy.size > max_points:
            pick = np.linspace(0, yy.size - 1, max_points).astype(int)
            yy = yy[pick]
            xx = xx[pick]
        moving_h, moving_w = int(moving_shape[0]), int(moving_shape[1])
        count = 0
        for x, y in zip(xx.astype(float), yy.astype(float)):
            mx, my = transform.TransformPoint((float(x), float(y)))
            if -0.5 <= mx <= moving_w - 0.5 and -0.5 <= my <= moving_h - 0.5:
                count += 1
        return count

    def prepare_affine_mi_inputs(
        fixed_img: np.ndarray,
        moving_img: np.ndarray,
        initial_moving_to_fixed_xy: np.ndarray,
    ) -> dict[str, Any]:
        import SimpleITK as sitk

        initial_xy = np.asarray(initial_moving_to_fixed_xy, dtype=float)
        fixed_to_moving_candidates = [
            ("stored affine inverse", np.linalg.inv(initial_xy)),
            ("stored affine direct", initial_xy),
        ]
        candidate_masks = []
        for label, fixed_to_moving_xy in fixed_to_moving_candidates:
            for candidate_label, candidate_fixed_to_moving_xy in (
                (f"{label} as fixed-to-moving", fixed_to_moving_xy),
                (f"{label} as moving-to-fixed", np.linalg.inv(fixed_to_moving_xy)),
            ):
                mask_arr = fixed_mask_valid_for_transform(
                    np.asarray(fixed_img).shape,
                    np.asarray(moving_img).shape,
                    candidate_fixed_to_moving_xy,
                )
                candidate_transform = sitk_affine_from_fixed_to_moving_matrix(candidate_fixed_to_moving_xy)
                candidate_masks.append(
                    (
                        candidate_label,
                        candidate_fixed_to_moving_xy,
                        candidate_transform,
                        mask_arr,
                        sitk_overlap_count_for_transform(candidate_transform, mask_arr, np.asarray(moving_img).shape),
                    )
                )
        transform_label, initial_fixed_to_moving_xy, _initial_transform, fixed_mask_arr, sitk_overlap_pixels = max(
            candidate_masks,
            key=lambda item: int(item[4]),
        )
        overlap_pixels = int(np.count_nonzero(fixed_mask_arr))
        if sitk_overlap_pixels < 8:
            raise ValueError(
                "The current MSI/reference affine does not produce enough SimpleITK fixed-to-moving overlap for optimization. "
                f"Overlap candidates: {', '.join(f'{label}={int(sitk_count)} SITK samples/{int(np.count_nonzero(mask))} mask pixels' for label, _matrix, _transform, mask, sitk_count in candidate_masks)}."
            )

        yy, xx = np.nonzero(fixed_mask_arr)
        fixed_h, fixed_w = np.asarray(fixed_img).shape
        pad = 16
        min_x = max(0, int(np.min(xx)) - pad)
        max_x = min(int(fixed_w) - 1, int(np.max(xx)) + pad)
        min_y = max(0, int(np.min(yy)) - pad)
        max_y = min(int(fixed_h) - 1, int(np.max(yy)) + pad)
        fixed_crop_arr = np.asarray(fixed_img)[min_y : max_y + 1, min_x : max_x + 1]
        fixed_crop_mask_arr = fixed_mask_arr[min_y : max_y + 1, min_x : max_x + 1]
        crop_to_full_xy = np.array(
            [[1.0, 0.0, float(min_x)], [0.0, 1.0, float(min_y)], [0.0, 0.0, 1.0]],
            dtype=float,
        )
        initial_crop_fixed_to_moving_xy = np.asarray(initial_fixed_to_moving_xy, dtype=float) @ crop_to_full_xy
        initial_transform = sitk_affine_from_fixed_to_moving_matrix(initial_crop_fixed_to_moving_xy)
        fixed_crop_sitk = sitk.GetImageFromArray(normalize_image_for_registration(fixed_crop_arr))
        fixed_crop_mask_sitk = sitk.GetImageFromArray(fixed_crop_mask_arr.astype(np.uint8, copy=False))
        fixed_crop_mask_sitk.CopyInformation(fixed_crop_sitk)
        moving_sitk = sitk.GetImageFromArray(normalize_image_for_registration(moving_img))
        moving_on_fixed_crop = sitk.Resample(
            moving_sitk,
            fixed_crop_sitk,
            initial_transform,
            sitk.sitkLinear,
            0.0,
            moving_sitk.GetPixelID(),
        )

        return {
            "fixed_crop_arr": fixed_crop_arr,
            "fixed_crop_mask_arr": fixed_crop_mask_arr,
            "fixed_crop_sitk": fixed_crop_sitk,
            "fixed_crop_mask_sitk": fixed_crop_mask_sitk,
            "moving_sitk": moving_sitk,
            "moving_on_fixed_crop_arr": sitk.GetArrayFromImage(moving_on_fixed_crop),
            "initial_transform": initial_transform,
            "crop_to_full_xy": crop_to_full_xy,
            "crop_bounds": (min_x, max_x, min_y, max_y),
            "transform_label": transform_label,
            "overlap_pixels": overlap_pixels,
            "sitk_overlap_pixels": int(sitk_overlap_pixels),
            "candidate_masks": candidate_masks,
        }

    def optimize_affine_with_mutual_information(
        fixed_img: np.ndarray,
        moving_img: np.ndarray,
        initial_moving_to_fixed_xy: np.ndarray,
        *,
        histogram_bins: int,
        learning_rate: float,
        min_step: float,
        iterations: int,
        sampling_percentage: float,
        seed: int,
        max_translation: float,
        max_linear_delta: float,
        max_passes: int,
        min_mi_improvement: float,
    ) -> tuple[np.ndarray, float, float, str, int, int, int]:
        import SimpleITK as sitk

        sitk.ProcessObject_SetGlobalWarningDisplay(False)
        current_transform_xy = np.asarray(initial_moving_to_fixed_xy, dtype=float).copy()
        first_before_mi = None
        last_after_mi = None
        last_transform_label = ""
        last_overlap_pixels = 0
        accepted_passes = 0
        evaluated_passes = 0

        for pass_idx in range(max(1, int(max_passes))):
            evaluated_passes = pass_idx + 1
            mi_inputs = prepare_affine_mi_inputs(fixed_img, moving_img, current_transform_xy)
            candidate_transform_xy, before_mi, after_mi, transform_label, overlap_pixels = optimize_affine_mi_single_pass(
                fixed_img,
                moving_img,
                mi_inputs=mi_inputs,
                histogram_bins=histogram_bins,
                learning_rate=learning_rate,
                min_step=min_step,
                iterations=iterations,
                max_translation=max_translation,
                max_linear_delta=max_linear_delta,
            )
            if first_before_mi is None:
                first_before_mi = before_mi
            improvement = float(after_mi - before_mi)
            if improvement < float(min_mi_improvement):
                last_after_mi = before_mi if last_after_mi is None else last_after_mi
                last_transform_label = f"{transform_label}; stopped at pass {pass_idx + 1}"
                last_overlap_pixels = overlap_pixels
                break
            current_transform_xy = np.asarray(candidate_transform_xy, dtype=float)
            last_after_mi = after_mi
            accepted_passes += 1
            last_transform_label = f"{transform_label}; {accepted_passes} accepted pass(es)"
            last_overlap_pixels = overlap_pixels

        return (
            current_transform_xy,
            float(first_before_mi),
            float(last_after_mi),
            last_transform_label,
            int(last_overlap_pixels),
            int(accepted_passes),
            int(evaluated_passes),
        )

    def optimize_affine_mi_single_pass(
        fixed_img: np.ndarray,
        moving_img: np.ndarray,
        *,
        mi_inputs: dict[str, Any],
        histogram_bins: int,
        learning_rate: float,
        min_step: float,
        iterations: int,
        max_translation: float,
        max_linear_delta: float,
    ) -> tuple[np.ndarray, float, float, str, int]:
        import SimpleITK as sitk

        fixed = mi_inputs["fixed_crop_sitk"]
        moving = sitk.GetImageFromArray(normalize_image_for_registration(mi_inputs["moving_on_fixed_crop_arr"]))
        moving.CopyInformation(fixed)
        initial_transform = sitk.AffineTransform(2)

        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=int(histogram_bins))
        registration.SetMetricFixedMask(mi_inputs["fixed_crop_mask_sitk"])
        registration.SetMetricSamplingStrategy(registration.NONE)
        registration.SetOptimizerAsRegularStepGradientDescent(
            float(learning_rate),
            float(min_step),
            int(iterations),
            0.5,
        )
        registration.SetOptimizerScales([1000.0, 1000.0, 1000.0, 1000.0, 1.0, 1.0])
        registration.SetInterpolator(sitk.sitkLinear)
        registration.SetShrinkFactorsPerLevel([2, 1])
        registration.SetSmoothingSigmasPerLevel([1, 0])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        registration.SetInitialTransform(initial_transform, inPlace=False)
        before_mi = -float(registration.MetricEvaluate(fixed, moving))

        try:
            final_transform = registration.Execute(fixed, moving)
        except Exception as exc:
            raise RuntimeError(
                "SimpleITK mutual-information optimization failed after initialization. "
                f"Chosen initializer: {mi_inputs['transform_label']}; fixed-mask overlap: {mi_inputs['overlap_pixels']} pixels; SITK overlap sample count: {mi_inputs['sitk_overlap_pixels']}. "
                f"Candidate overlaps: {', '.join(f'{label}={int(sitk_count)} SITK samples/{int(np.count_nonzero(mask))} mask pixels' for label, _matrix, _transform, mask, sitk_count in mi_inputs['candidate_masks'])}. "
                f"Fixed shape: {np.asarray(fixed_img).shape}; fixed crop: {mi_inputs['fixed_crop_arr'].shape} at x={mi_inputs['crop_bounds'][0]}:{mi_inputs['crop_bounds'][1]}, y={mi_inputs['crop_bounds'][2]}:{mi_inputs['crop_bounds'][3]}; moving shape: {np.asarray(moving_img).shape}. "
                f"Original error: {exc}"
            ) from exc
        delta_crop_fixed_to_prewarped_xy = sitk_transform_to_homogeneous_matrix(final_transform)
        linear_delta = float(np.linalg.norm(delta_crop_fixed_to_prewarped_xy[:2, :2] - np.eye(2), ord="fro"))
        translation_delta = float(np.linalg.norm(delta_crop_fixed_to_prewarped_xy[:2, 2]))
        if translation_delta > float(max_translation) or linear_delta > float(max_linear_delta):
            raise RuntimeError(
                "Optimization produced a larger-than-allowed affine delta and was rejected. "
                f"Translation delta: {translation_delta:.3g} px (limit {float(max_translation):.3g}); "
                f"linear/shear delta: {linear_delta:.3g} (limit {float(max_linear_delta):.3g}). "
                "Try a lower learning rate, fewer iterations, or a looser cap if the preview still looks reasonable."
            )
        initial_crop_fixed_to_moving_xy = sitk_transform_to_homogeneous_matrix(mi_inputs["initial_transform"])
        crop_fixed_to_moving_xy = initial_crop_fixed_to_moving_xy @ delta_crop_fixed_to_prewarped_xy
        fixed_to_moving_xy = crop_fixed_to_moving_xy @ np.linalg.inv(mi_inputs["crop_to_full_xy"])
        moving_to_fixed_xy = np.linalg.inv(fixed_to_moving_xy)
        after_mi = -float(registration.GetMetricValue())
        return moving_to_fixed_xy, before_mi, after_mi, mi_inputs["transform_label"], mi_inputs["overlap_pixels"]

    def show_affine_optimization_result_dialog(
        state: dict[str, Any],
        fixed_img: np.ndarray,
        moving_img: np.ndarray,
        candidate_transform_xy: np.ndarray,
        before_mi: float,
        after_mi: float,
        transform_label: str,
        overlap_pixels: int,
        accepted_passes: int,
        evaluated_passes: int,
    ):
        try:
            before_inputs = prepare_affine_mi_inputs(fixed_img, moving_img, state["current_transform_xy"])
            fixed_crop = normalize_image_for_registration(before_inputs["fixed_crop_arr"])
            before_overlay = normalize_image_for_registration(before_inputs["moving_on_fixed_crop_arr"])
            after_overlay = normalize_image_for_registration(
                resample_moving_on_fixed_crop(
                    moving_img,
                    before_inputs["fixed_crop_sitk"],
                    before_inputs["crop_to_full_xy"],
                    candidate_transform_xy,
                )
            )
        except Exception as exc:
            QMessageBox.warning(None, "Affine Optimization Preview", str(exc))
            return

        dialog = QDialog()
        dialog.setWindowTitle("Review Optimized Affine")
        dialog.setModal(False)
        dialog.resize(980, 560)
        layout = QVBoxLayout(dialog)
        fig = Figure(figsize=(9.8, 4.8), constrained_layout=True)
        canvas = FigureCanvas(fig)
        axes = fig.subplots(1, 2)
        for ax in np.ravel(axes):
            ax.set_axis_off()
            ax.imshow(fixed_crop, cmap="gray")
        before_masked = np.ma.masked_where(before_overlay <= 0, before_overlay)
        after_masked = np.ma.masked_where(after_overlay <= 0, after_overlay)
        axes[0].imshow(before_masked, cmap="magma", alpha=0.55)
        axes[0].set_title("Before optimization", loc="left")
        axes[1].imshow(after_masked, cmap="magma", alpha=0.55)
        axes[1].set_title("After optimization", loc="left")
        min_x, max_x, min_y, max_y = before_inputs["crop_bounds"]
        fig.suptitle(
            f"Mutual information: before {before_mi:.6g}, after {after_mi:.6g}; "
            f"passes accepted {accepted_passes}/{evaluated_passes}; crop x={min_x}:{max_x}, y={min_y}:{max_y}"
        )
        layout.addWidget(canvas)

        button_row = QWidget()
        button_layout = QGridLayout(button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        reject_button = QPushButton("Reject")
        accept_button = QPushButton("Accept")
        button_layout.addWidget(reject_button, 0, 0)
        button_layout.addWidget(accept_button, 0, 1)
        layout.addWidget(button_row)

        def accept_candidate():
            state["current_transform_xy"][:] = np.asarray(candidate_transform_xy, dtype=float)
            remove_optimization_preview(state, restore_visibility=False)
            state["ion_layer"].visible = True
            apply_transform_to_state(state)
            sync_controls_to_active_dataset()
            dialog.accept()

        def reject_candidate():
            remove_optimization_preview(state, restore_visibility=True)
            apply_transform_to_state(state)
            sync_controls_to_active_dataset()
            dialog.reject()

        accept_button.clicked.connect(accept_candidate)
        reject_button.clicked.connect(reject_candidate)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        if not hasattr(viewer, "_viu_chem_debug_dialogs"):
            viewer._viu_chem_debug_dialogs = []
        viewer._viu_chem_debug_dialogs.append(dialog)

    @magicgui(
        reference_channel={"widget_type": "ComboBox", "choices": ["(none)"]},
        call_button="Preview MI Inputs",
    )
    def preview_affine_mi_inputs_widget(reference_channel: str = "(none)"):
        state = get_active_state()
        layer = _get_reference_layer_by_name(reference_channel)
        if layer is None:
            QMessageBox.warning(None, "Preview MI Inputs", "Select a fluorescence/reference channel first.")
            return
        def prepare_preview():
            fixed_img = _reference_intensity_from_layer(layer)
            moving_img = active_msi_registration_image(state)
            mi_inputs = prepare_affine_mi_inputs(fixed_img, moving_img, state["current_transform_xy"])
            return fixed_img, moving_img, mi_inputs

        try:
            fixed_img, moving_img, mi_inputs = _run_with_busy_dialog(
                "Preview MI Inputs",
                "Preparing mutual-information input preview...",
                prepare_preview,
            )
        except ModuleNotFoundError:
            QMessageBox.warning(None, "Preview MI Inputs", "SimpleITK is not installed. Install the coregistration extra again to enable this tool.")
            return
        except Exception as exc:
            QMessageBox.warning(None, "Preview MI Inputs", str(exc))
            return

        fixed_crop = normalize_image_for_registration(mi_inputs["fixed_crop_arr"])
        moving_overlay = normalize_image_for_registration(mi_inputs["moving_on_fixed_crop_arr"])
        dialog = QDialog()
        dialog.setWindowTitle("MI Input Preview")
        dialog.setModal(False)
        dialog.resize(920, 460)
        layout = QVBoxLayout(dialog)
        fig = Figure(figsize=(9.2, 4.6), constrained_layout=True)
        canvas = FigureCanvas(fig)
        axes = fig.subplots(1, 2)
        for ax in np.ravel(axes):
            ax.set_axis_off()
        axes[0].imshow(fixed_crop, cmap="gray")
        axes[0].set_title("Fixed reference crop", loc="left")
        axes[1].imshow(fixed_crop, cmap="gray")
        overlay = np.ma.masked_where(moving_overlay <= 0, moving_overlay)
        axes[1].imshow(overlay, cmap="magma", alpha=0.55)
        min_x, max_x, min_y, max_y = mi_inputs["crop_bounds"]
        axes[1].set_title(
            f"MSI after current affine\n{mi_inputs['transform_label']}; overlap {mi_inputs['sitk_overlap_pixels']} samples",
            loc="left",
        )
        fig.suptitle(f"Crop x={min_x}:{max_x}, y={min_y}:{max_y}; moving shape {np.asarray(moving_img).shape}")
        layout.addWidget(canvas)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        if not hasattr(viewer, "_viu_chem_debug_dialogs"):
            viewer._viu_chem_debug_dialogs = []
        viewer._viu_chem_debug_dialogs.append(dialog)

    @magicgui(
        reference_channel={"widget_type": "ComboBox", "choices": ["(none)"]},
        histogram_bins={"widget_type": "SpinBox", "min": 8, "max": 256, "step": 1},
        learning_rate={"widget_type": "FloatSpinBox", "min": 0.0001, "max": 100.0, "step": 0.1},
        min_step={"widget_type": "FloatSpinBox", "min": 1e-8, "max": 1.0, "step": 1e-4},
        iterations={"widget_type": "SpinBox", "min": 1, "max": 5000, "step": 25},
        sampling_percentage={"widget_type": "FloatSpinBox", "min": 0.001, "max": 1.0, "step": 0.05},
        seed={"widget_type": "SpinBox", "min": 0, "max": 1000000, "step": 1},
        max_translation={"widget_type": "FloatSpinBox", "min": 0.0, "max": 1000.0, "step": 1.0},
        max_linear_delta={"widget_type": "FloatSpinBox", "min": 0.0, "max": 10.0, "step": 0.01},
        max_passes={"widget_type": "SpinBox", "min": 1, "max": 100, "step": 1},
        min_mi_improvement={"widget_type": "FloatSpinBox", "min": 0.0, "max": 10.0, "step": 0.0001},
        call_button="Optimize Affine With MI",
    )
    def optimize_affine_registration_widget(
        reference_channel: str = "(none)",
        histogram_bins: int = 50,
        learning_rate: float = 0.05,
        min_step: float = 1e-4,
        iterations: int = 150,
        sampling_percentage: float = 0.2,
        seed: int = 42,
        max_translation: float = 25.0,
        max_linear_delta: float = 0.15,
        max_passes: int = 25,
        min_mi_improvement: float = 0.001,
    ):
        state = get_active_state()
        layer = _get_reference_layer_by_name(reference_channel)
        if layer is None:
            QMessageBox.warning(None, "Affine Optimization", "Select a fluorescence/reference channel first.")
            return
        def run_optimization():
            fixed_img = _reference_intensity_from_layer(layer)
            moving_img = active_msi_registration_image(state)
            result = optimize_affine_with_mutual_information(
                fixed_img,
                moving_img,
                state["current_transform_xy"],
                histogram_bins=int(histogram_bins),
                learning_rate=float(learning_rate),
                min_step=float(min_step),
                iterations=int(iterations),
                sampling_percentage=float(sampling_percentage),
                seed=int(seed),
                max_translation=float(max_translation),
                max_linear_delta=float(max_linear_delta),
                max_passes=int(max_passes),
                min_mi_improvement=float(min_mi_improvement),
            )
            return fixed_img, moving_img, result

        try:
            fixed_img, moving_img, optimization_result = _run_with_busy_dialog(
                "Affine Optimization",
                "Optimizing affine registration with mutual information...\nThis can take a minute.",
                run_optimization,
            )
            (
                candidate_transform_xy,
                before_mi,
                after_mi,
                transform_label,
                overlap_pixels,
                accepted_passes,
                evaluated_passes,
            ) = optimization_result
        except ModuleNotFoundError:
            QMessageBox.warning(None, "Affine Optimization", "SimpleITK is not installed. Install the coregistration extra again to enable this tool.")
            return
        except Exception as exc:
            QMessageBox.warning(None, "Affine Optimization", str(exc))
            return

        show_affine_optimization_result_dialog(
            state,
            fixed_img,
            moving_img,
            candidate_transform_xy,
            before_mi,
            after_mi,
            transform_label,
            overlap_pixels,
            accepted_passes,
            evaluated_passes,
        )

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
    if_layer_controls_layout = QVBoxLayout(if_layer_controls)
    if_layer_controls_layout.setContentsMargins(0, 0, 0, 0)
    if_layer_controls_layout.setSpacing(6)
    if_layer_table = QTableWidget()
    if_layer_table.setColumnCount(7)
    if_layer_table.setHorizontalHeaderLabels(["Show", "Name", "Color", "Mode", "Low", "High", "Gamma"])
    if_layer_table.setAlternatingRowColors(True)
    if_layer_table.setSortingEnabled(False)
    try:
        if_layer_table.horizontalHeader().setStretchLastSection(True)
    except Exception:
        pass
    if_layer_table.setMinimumHeight(320)
    if_apply_table_button = QPushButton("Save IF Settings to Zarr")
    if_export_csv_button = QPushButton("Export IF Settings CSV")
    if_import_csv_button = QPushButton("Import IF Settings CSV")
    if_button_row = QWidget()
    if_button_row_layout = QGridLayout(if_button_row)
    if_button_row_layout.setContentsMargins(0, 0, 0, 0)
    if_button_row_layout.addWidget(if_apply_table_button, 0, 0)
    if_button_row_layout.addWidget(if_export_csv_button, 0, 1)
    if_button_row_layout.addWidget(if_import_csv_button, 1, 0, 1, 2)
    if_layer_controls_layout.addWidget(if_layer_table)
    if_layer_controls_layout.addWidget(if_button_row)
    suppress_if_table_updates = False

    def _configure_if_contrast_spin(spin: QDoubleSpinBox, mode: str):
        if str(mode).lower() == "percentile":
            spin.setRange(0.0, 100.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.1)
        else:
            spin.setRange(-1e15, 1e15)
            spin.setDecimals(4)
            spin.setSingleStep(1.0)

    def _if_table_contrast_values(layer) -> tuple[str, float, float]:
        metadata = _layer_metadata(layer)
        mode = str(metadata.get("reference_contrast_mode", "percentile"))
        if mode == "intensity":
            limits = tuple(float(v) for v in getattr(layer, "contrast_limits", finite_data_limits(np.asarray(layer.data))))
            return mode, limits[0], limits[1]
        low, high = tuple(float(v) for v in metadata.get("reference_contrast_percentiles", (1.0, 99.8)))
        return "percentile", low, high

    def apply_if_table_row(row_idx: int):
        if suppress_if_table_updates:
            return
        name_item = if_layer_table.item(row_idx, 1)
        if name_item is None:
            return
        layer = name_item.data(Qt.UserRole)
        if layer is None:
            return
        show_box = if_layer_table.cellWidget(row_idx, 0)
        color_combo = if_layer_table.cellWidget(row_idx, 2)
        mode_combo = if_layer_table.cellWidget(row_idx, 3)
        low_spin = if_layer_table.cellWidget(row_idx, 4)
        high_spin = if_layer_table.cellWidget(row_idx, 5)
        gamma_spin = if_layer_table.cellWidget(row_idx, 6)
        layer.visible = bool(show_box.isChecked()) if isinstance(show_box, QCheckBox) else bool(layer.visible)
        layer.opacity = 1.0
        layer.blending = "translucent"
        new_name = str(name_item.text()).strip()
        if new_name:
            layer.name = new_name
        if isinstance(color_combo, QComboBox):
            _set_reference_layer_color(layer, str(color_combo.currentText()))
        mode = str(mode_combo.currentText()) if isinstance(mode_combo, QComboBox) else "percentile"
        low = float(low_spin.value()) if isinstance(low_spin, QDoubleSpinBox) else 0.0
        high = float(high_spin.value()) if isinstance(high_spin, QDoubleSpinBox) else 1.0
        try:
            if mode == "intensity":
                _apply_reference_layer_contrast(layer, mode, intensity_limits=(low, high))
            else:
                _apply_reference_layer_contrast(layer, mode, percentiles=(low, high))
        except Exception:
            pass
        if isinstance(gamma_spin, QDoubleSpinBox):
            _apply_reference_layer_gamma(layer, float(gamma_spin.value()))

    def rebuild_if_layer_controls():
        nonlocal suppress_if_table_updates
        suppress_if_table_updates = True
        layers = _reference_channel_layers()
        if not layers:
            if_layer_table.setRowCount(0)
            suppress_if_table_updates = False
            return

        if_layer_table.setRowCount(len(layers))
        for row_idx, layer in enumerate(layers):
            metadata = _layer_metadata(layer)
            show_box = QCheckBox()
            show_box.setChecked(bool(layer.visible))
            name_item = QTableWidgetItem(str(layer.name))
            name_item.setData(Qt.UserRole, layer)
            color_combo = QComboBox()
            color_combo.addItems(_reference_color_choices())
            color_choice = str(metadata.get("reference_color_choice", "metadata"))
            color_combo.setCurrentText(color_choice if color_choice in _reference_color_choices() else "metadata")
            mode, low, high = _if_table_contrast_values(layer)
            mode_combo = QComboBox()
            mode_combo.addItems(["percentile", "intensity"])
            mode_combo.setCurrentText(mode)
            low_spin = QDoubleSpinBox()
            high_spin = QDoubleSpinBox()
            gamma_spin = QDoubleSpinBox()
            _configure_if_contrast_spin(low_spin, mode)
            _configure_if_contrast_spin(high_spin, mode)
            low_spin.setValue(float(low))
            high_spin.setValue(float(high))
            gamma_spin.setRange(0.01, 10.0)
            gamma_spin.setDecimals(3)
            gamma_spin.setSingleStep(0.05)
            gamma_spin.setValue(float(metadata.get("reference_gamma", getattr(layer, "gamma", 1.0))))

            def _set_mode(value, layer=layer, low_spin=low_spin, high_spin=high_spin):
                mode = str(value)
                _configure_if_contrast_spin(low_spin, mode)
                _configure_if_contrast_spin(high_spin, mode)
                if mode == "intensity":
                    limits = tuple(float(v) for v in getattr(layer, "contrast_limits", finite_data_limits(np.asarray(layer.data))))
                    low_spin.setValue(limits[0])
                    high_spin.setValue(limits[1])
                else:
                    metadata = _layer_metadata(layer)
                    low, high = tuple(float(v) for v in metadata.get("reference_contrast_percentiles", (1.0, 99.8)))
                    low_spin.setValue(low)
                    high_spin.setValue(high)

            mode_combo.currentTextChanged.connect(_set_mode)
            show_box.toggled.connect(lambda _checked, row_idx=row_idx: apply_if_table_row(row_idx))
            color_combo.currentTextChanged.connect(lambda _value, row_idx=row_idx: apply_if_table_row(row_idx))
            mode_combo.currentTextChanged.connect(lambda _value, row_idx=row_idx: apply_if_table_row(row_idx))
            low_spin.valueChanged.connect(lambda _value, row_idx=row_idx: apply_if_table_row(row_idx))
            high_spin.valueChanged.connect(lambda _value, row_idx=row_idx: apply_if_table_row(row_idx))
            gamma_spin.valueChanged.connect(lambda _value, row_idx=row_idx: apply_if_table_row(row_idx))

            if_layer_table.setCellWidget(row_idx, 0, show_box)
            if_layer_table.setItem(row_idx, 1, name_item)
            if_layer_table.setCellWidget(row_idx, 2, color_combo)
            if_layer_table.setCellWidget(row_idx, 3, mode_combo)
            if_layer_table.setCellWidget(row_idx, 4, low_spin)
            if_layer_table.setCellWidget(row_idx, 5, high_spin)
            if_layer_table.setCellWidget(row_idx, 6, gamma_spin)
        if_layer_table.resizeColumnsToContents()
        suppress_if_table_updates = False

    def on_if_table_item_changed(item: QTableWidgetItem):
        if item.column() == 1:
            apply_if_table_row(item.row())

    if_layer_table.itemChanged.connect(on_if_table_item_changed)

    def save_if_table_settings():
        for row_idx in range(if_layer_table.rowCount()):
            apply_if_table_row(row_idx)
        current_threshold_layer = _get_reference_layer_by_name(str(if_threshold_reference_channel.value))
        if current_threshold_layer is not None:
            metadata = _layer_metadata(current_threshold_layer)
            metadata["if_threshold_percentile"] = float(if_threshold_percentile.value)
            metadata["if_threshold_value"] = float(if_threshold_absolute_value.value)
            metadata["if_threshold_prefilter_annotation"] = str(if_threshold_prefilter_annotation.value)
            metadata["if_threshold_prefilter_region"] = str(if_threshold_prefilter_region.value)
        layers_by_key: dict[str, list[Any]] = {}
        for layer in _reference_channel_layers():
            reference_key = str(_layer_metadata(layer).get("reference_key", ""))
            if reference_key:
                layers_by_key.setdefault(reference_key, []).append(layer)
        settings_by_key: dict[str, dict[str, dict[str, Any]]] = {}
        for layer in _reference_channel_layers():
            metadata = _layer_metadata(layer)
            reference_key = str(metadata.get("reference_key", ""))
            if not reference_key:
                continue
            channel_index = str(int(metadata.get("reference_channel_index", 0)))
            settings_by_key.setdefault(reference_key, {})[channel_index] = {
                "display_name": str(layer.name),
                "visible": bool(layer.visible),
                "color_choice": str(metadata.get("reference_color_choice", "metadata")),
                "contrast_mode": str(metadata.get("reference_contrast_mode", "percentile")),
                "contrast_percentiles": [float(v) for v in metadata.get("reference_contrast_percentiles", (1.0, 99.8))],
                "contrast_limits": [float(v) for v in getattr(layer, "contrast_limits", metadata.get("reference_contrast_limits", (0.0, 1.0)))],
                "gamma": float(metadata.get("reference_gamma", getattr(layer, "gamma", 1.0))),
                "if_threshold_percentile": metadata.get("if_threshold_percentile", ""),
                "if_threshold_value": metadata.get("if_threshold_value", ""),
                "if_threshold_prefilter_annotation": str(metadata.get("if_threshold_prefilter_annotation", "")),
                "if_threshold_prefilter_region": str(metadata.get("if_threshold_prefilter_region", "")),
            }
        try:
            root = zarr.open_group(host_zarr_path, mode="r+", use_consolidated=False)
            for reference_key, saved_settings in settings_by_key.items():
                if "images" not in root or reference_key not in root["images"]:
                    continue
                root["images"][reference_key].attrs["if_display_settings"] = saved_settings
                for state in datasets.values():
                    sdata = state["dataset"].sdata
                    if reference_key in sdata.images:
                        sdata.images[reference_key].attrs["if_display_settings"] = saved_settings
            QMessageBox.information(None, "IF Display Settings", "Saved IF display settings to the zarr.")
        except Exception as exc:
            QMessageBox.warning(None, "IF Display Settings", f"Could not save IF display settings:\n{exc}")

    if_settings_csv_columns = [
        "reference_key",
        "channel_index",
        "display_name",
        "visible",
        "color_choice",
        "contrast_mode",
        "contrast_low",
        "contrast_high",
        "contrast_limit_low",
        "contrast_limit_high",
        "gamma",
        "if_threshold_percentile",
        "if_threshold_value",
        "if_threshold_prefilter_annotation",
        "if_threshold_prefilter_region",
    ]

    def _parse_csv_bool(value: Any, default: bool = True) -> bool:
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        return bool(default)

    def _parse_csv_float(value: Any, default: float | None = None) -> float | None:
        text = str(value).strip()
        if text == "":
            return default
        try:
            out = float(text)
        except Exception:
            return default
        return out if np.isfinite(out) else default

    def export_if_settings_csv():
        for row_idx in range(if_layer_table.rowCount()):
            apply_if_table_row(row_idx)
        current_threshold_layer = _get_reference_layer_by_name(str(if_threshold_reference_channel.value))
        if current_threshold_layer is not None:
            metadata = _layer_metadata(current_threshold_layer)
            metadata["if_threshold_percentile"] = float(if_threshold_percentile.value)
            metadata["if_threshold_value"] = float(if_threshold_absolute_value.value)
            metadata["if_threshold_prefilter_annotation"] = str(if_threshold_prefilter_annotation.value)
            metadata["if_threshold_prefilter_region"] = str(if_threshold_prefilter_region.value)

        dialog = QFileDialog(None, "Export IF settings CSV")
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setFileMode(QFileDialog.FileMode.AnyFile)
        dialog.setNameFilter("CSV (*.csv)")
        dialog.setDirectory(str(host_zarr_path.parent))
        dialog.selectFile("if_settings.csv")
        if dialog.exec() != QFileDialog.DialogCode.Accepted:
            return
        selected = dialog.selectedFiles()
        if not selected:
            return
        path = Path(selected[0]).expanduser()
        if path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")

        def write_csv():
            rows = []
            for layer in _reference_channel_layers():
                metadata = _layer_metadata(layer)
                reference_key = str(metadata.get("reference_key", ""))
                if not reference_key:
                    continue
                channel_index = int(metadata.get("reference_channel_index", 0))
                mode = str(metadata.get("reference_contrast_mode", "percentile"))
                pct_low, pct_high = tuple(float(v) for v in metadata.get("reference_contrast_percentiles", (1.0, 99.8)))
                limit_low, limit_high = tuple(float(v) for v in getattr(layer, "contrast_limits", metadata.get("reference_contrast_limits", (0.0, 1.0))))
                rows.append(
                    {
                        "reference_key": reference_key,
                        "channel_index": channel_index,
                        "display_name": str(layer.name),
                        "visible": bool(layer.visible),
                        "color_choice": str(metadata.get("reference_color_choice", "metadata")),
                        "contrast_mode": mode,
                        "contrast_low": pct_low,
                        "contrast_high": pct_high,
                        "contrast_limit_low": limit_low,
                        "contrast_limit_high": limit_high,
                        "gamma": float(metadata.get("reference_gamma", getattr(layer, "gamma", 1.0))),
                        "if_threshold_percentile": metadata.get("if_threshold_percentile", ""),
                        "if_threshold_value": metadata.get("if_threshold_value", ""),
                        "if_threshold_prefilter_annotation": str(metadata.get("if_threshold_prefilter_annotation", "")),
                        "if_threshold_prefilter_region": str(metadata.get("if_threshold_prefilter_region", "")),
                    }
                )
            with path.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=if_settings_csv_columns)
                writer.writeheader()
                writer.writerows(rows)

        try:
            _run_with_busy_dialog("Export IF Settings CSV", "Exporting IF settings CSV...", write_csv)
        except Exception as exc:
            QMessageBox.warning(None, "Export IF Settings CSV", str(exc))

    def import_if_settings_csv():
        path, _ = QFileDialog.getOpenFileName(None, "Import IF settings CSV", str(host_zarr_path.parent), "CSV (*.csv);;All files (*)")
        if not path:
            return
        csv_path = Path(path).expanduser()

        def read_rows():
            with csv_path.open("r", newline="") as fh:
                return list(csv.DictReader(fh))

        try:
            rows = _run_with_busy_dialog("Import IF Settings CSV", "Importing IF settings CSV...", read_rows)
        except Exception as exc:
            QMessageBox.warning(None, "Import IF Settings CSV", str(exc))
            return

        matched = 0
        for row in rows:
            reference_key = str(row.get("reference_key", "")).strip()
            channel_index = _parse_csv_float(row.get("channel_index", ""), None)
            layer = _get_reference_layer_by_key_channel(reference_key, int(channel_index)) if channel_index is not None else None
            if layer is None:
                display_name = str(row.get("display_name", "")).strip()
                layer = _get_reference_layer_by_name(display_name) if display_name else None
            if layer is None:
                continue
            matched += 1
            metadata = _layer_metadata(layer)
            display_name = str(row.get("display_name", "")).strip()
            if display_name:
                layer.name = display_name
            layer.visible = _parse_csv_bool(row.get("visible", layer.visible), bool(layer.visible))
            color_choice = str(row.get("color_choice", metadata.get("reference_color_choice", "metadata"))).strip() or "metadata"
            _set_reference_layer_color(layer, color_choice)
            mode = str(row.get("contrast_mode", metadata.get("reference_contrast_mode", "percentile"))).strip().lower() or "percentile"
            pct_low = _parse_csv_float(row.get("contrast_low", ""), 1.0)
            pct_high = _parse_csv_float(row.get("contrast_high", ""), 99.8)
            limit_low = _parse_csv_float(row.get("contrast_limit_low", ""), None)
            limit_high = _parse_csv_float(row.get("contrast_limit_high", ""), None)
            if mode == "intensity" and limit_low is not None and limit_high is not None:
                _apply_reference_layer_contrast(layer, "intensity", intensity_limits=(float(limit_low), float(limit_high)))
            else:
                _apply_reference_layer_contrast(layer, "percentile", percentiles=(float(pct_low), float(pct_high)))
            gamma = _parse_csv_float(row.get("gamma", ""), float(metadata.get("reference_gamma", getattr(layer, "gamma", 1.0))))
            _apply_reference_layer_gamma(layer, float(gamma if gamma is not None else 1.0))
            threshold_percentile = _parse_csv_float(row.get("if_threshold_percentile", ""), None)
            threshold_value = _parse_csv_float(row.get("if_threshold_value", ""), None)
            if threshold_percentile is not None:
                metadata["if_threshold_percentile"] = float(threshold_percentile)
            if threshold_value is not None:
                metadata["if_threshold_value"] = float(threshold_value)
            threshold_annotation = str(row.get("if_threshold_prefilter_annotation", "")).strip()
            threshold_region = str(row.get("if_threshold_prefilter_region", "")).strip()
            if threshold_annotation:
                metadata["if_threshold_prefilter_annotation"] = threshold_annotation
            if threshold_region:
                metadata["if_threshold_prefilter_region"] = threshold_region

        rebuild_if_layer_controls()
        refresh_if_threshold_choices(get_active_state())
        QMessageBox.information(None, "Import IF Settings CSV", f"Imported settings for {matched} IF channel(s).")

    if_apply_table_button.clicked.connect(save_if_table_settings)
    if_export_csv_button.clicked.connect(export_if_settings_csv)
    if_import_csv_button.clicked.connect(import_if_settings_csv)

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
        ion_display_options.contrast_mode.value = str(state.get("current_contrast_mode", "percentile"))
        ion_display_options.contrast_percentiles.value = (
            float(state["current_contrast_low_pct"]),
            float(state["current_contrast_high_pct"]),
        )
        ion_display_options.absolute_low.value = f"{float(state.get('current_contrast_low', 0.0)):g}"
        ion_display_options.absolute_high.value = f"{float(state.get('current_contrast_high', 1.0)):g}"
        mz_selector.target_mz.value = f"{float(state['current_target_mz']):.4f}"
        mz_selector.ppm_tolerance.value = float(state["current_ppm_tolerance"])
        threshold_map_to_dataset.choices = ordered_choices if ordered_choices else [dataset_choice_text(state)]
        threshold_map_to_dataset.value = dataset_choice_text(state)
        roi_shape_keys = [key for key in coreg_dataset.sdata.shapes.keys() if "pixels" not in key.lower()]
        roi_mask_controls.roi_shape_key.choices = roi_shape_keys if roi_shape_keys else ["(none)"]
        if roi_mask_controls.roi_shape_key.value not in roi_mask_controls.roi_shape_key.choices:
            roi_mask_controls.roi_shape_key.value = roi_mask_controls.roi_shape_key.choices[0]
        roi_label_choices = annotation_region_choices(coreg_dataset, str(roi_mask_controls.roi_shape_key.value))
        roi_mask_controls.region_label.choices = roi_label_choices
        if roi_mask_controls.region_label.value not in roi_label_choices:
            roi_mask_controls.region_label.value = roi_label_choices[0]
        refresh_threshold_prefilter_choices(state)
        refresh_if_threshold_choices(state)
        refresh_annotation_widget_choices()
        rebuild_msi_layer_controls()
        _refresh_if_toolbox_widgets()
        preview_affine_mi_inputs_widget.reference_channel.choices = _reference_channel_choice_names()
        if preview_affine_mi_inputs_widget.reference_channel.value not in preview_affine_mi_inputs_widget.reference_channel.choices:
            preview_affine_mi_inputs_widget.reference_channel.value = preview_affine_mi_inputs_widget.reference_channel.choices[0]
        optimize_affine_registration_widget.reference_channel.choices = _reference_channel_choice_names()
        if optimize_affine_registration_widget.reference_channel.value not in optimize_affine_registration_widget.reference_channel.choices:
            optimize_affine_registration_widget.reference_channel.value = optimize_affine_registration_widget.reference_channel.choices[0]
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
                threshold_preview_layer = other_state.get("threshold_preview_layer")
                if threshold_preview_layer is not None:
                    threshold_preview_layer.visible = False
                optimization_preview_layer = other_state.get("optimization_preview_layer")
                if optimization_preview_layer is not None:
                    optimization_preview_layer.visible = False
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
        embedded = _run_with_busy_dialog(
            "Add MSI Dataset",
            "Importing MSI dataset...\nThis can take a little while.",
            lambda: embed_msi_dataset(host_zarr_path, picked_path, registered_cs=registered_cs),
        )
        new_dataset = CoregistrationDataset(host_zarr_path, registered_cs=registered_cs, table_key=embedded["table_key"], tic_key=embedded["tic_key"])
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
        _run_with_busy_dialog(
            "Add Optical Image",
            "Importing optical image...",
            lambda: add_reference_image(coreg_dataset.zarr_path, path, key="optical", registered_cs=registered_cs),
        )
        coreg_dataset.sdata = _run_with_busy_dialog(
            "Add Optical Image",
            "Refreshing optical image layers...",
            lambda: sd.read_zarr(coreg_dataset.zarr_path),
        )
        if "optical" not in coreg_dataset.reference_image_keys:
            coreg_dataset.reference_image_keys.append("optical")
        add_or_update_reference_layer(coreg_dataset, "optical")
        refresh_if_threshold_choices(state)

    @magicgui(call_button="Add/Update H&E")
    def add_hne_image():
        state = get_active_state()
        coreg_dataset = state["dataset"]
        path, _ = QFileDialog.getOpenFileName(None, "Select H&E image", "", "Image files (*.tif *.tiff *.png *.jpg *.jpeg);;All files (*)")
        if not path:
            return
        _run_with_busy_dialog(
            "Add H&E Image",
            "Importing H&E image...",
            lambda: add_reference_image(coreg_dataset.zarr_path, path, key="hne", registered_cs=registered_cs),
        )
        coreg_dataset.sdata = _run_with_busy_dialog(
            "Add H&E Image",
            "Refreshing H&E image layers...",
            lambda: sd.read_zarr(coreg_dataset.zarr_path),
        )
        if "hne" not in coreg_dataset.reference_image_keys:
            coreg_dataset.reference_image_keys.append("hne")
        add_or_update_reference_layer(coreg_dataset, "hne")
        refresh_if_threshold_choices(state)

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
        _run_with_busy_dialog(
            "Add H&E QPTIFF",
            "Importing H&E QPTIFF...\nLarge pyramid images can take a little while.",
            lambda: add_reference_image(coreg_dataset.zarr_path, path, key="hne", registered_cs=registered_cs, qptiff_level=int(qptiff_level)),
        )
        coreg_dataset.sdata = _run_with_busy_dialog(
            "Add H&E QPTIFF",
            "Refreshing H&E image layers...",
            lambda: sd.read_zarr(coreg_dataset.zarr_path),
        )
        if "hne" not in coreg_dataset.reference_image_keys:
            coreg_dataset.reference_image_keys.append("hne")
        add_or_update_reference_layer(coreg_dataset, "hne")
        refresh_if_threshold_choices(state)

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
        keys = _run_with_busy_dialog(
            "Add GeoJSON",
            "Importing GeoJSON annotations...",
            lambda: import_geojson_annotations(
                coreg_dataset.zarr_path,
                paths,
                target_image=target_image,
                name_prefix=name_prefix,
                registered_cs=registered_cs,
                object_mode=object_mode,
                max_shapes=int(max_shapes),
                simplify_tolerance=float(simplify_tolerance),
                annotation_pyramid_level=(int(annotation_pyramid_level) if int(annotation_pyramid_level) >= 0 else None),
            ),
        )
        def refresh_geojson_layers():
            coreg_dataset.sdata = sd.read_zarr(coreg_dataset.zarr_path)
            add_annotation_shape_layers(state, keys)
        _run_with_busy_dialog("Add GeoJSON", "Refreshing annotation layers...", refresh_geojson_layers)
        sync_controls_to_active_dataset()

    @magicgui(
        annotation_key={"widget_type": "ComboBox", "choices": ["(none)"]},
        call_button="Remove Annotation",
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
        rewritten = _run_with_busy_dialog(
            "Adjust GeoJSON",
            "Updating GeoJSON annotation geometry...",
            lambda: transform_geojson_annotations(
                state["dataset"].zarr_path,
                [annotation_key],
                annotation_scale_x=sx,
                annotation_scale_y=sy,
                annotation_translate_x=tx,
                annotation_translate_y=ty,
            ),
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
        _run_with_busy_dialog(
            "Save Registration",
            "Saving active registration...",
            lambda: save_coregistration(
                state["dataset"].zarr_path,
                state["current_transform_xy"],
                table_key=state["dataset"].table_key,
                tic_key=state["dataset"].tic_key,
                registered_cs=registered_cs,
            ),
        )

    @magicgui(call_button="Save All Registrations")
    def save_all_registrations_widget():
        def save_all():
            for state in datasets.values():
                save_coregistration(
                    state["dataset"].zarr_path,
                    state["current_transform_xy"],
                    table_key=state["dataset"].table_key,
                    tic_key=state["dataset"].tic_key,
                    registered_cs=registered_cs,
                )
        _run_with_busy_dialog("Save Registrations", "Saving all registrations...", save_all)

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
        def export_tiff():
            screenshot = viewer.screenshot(canvas_only=True, flash=False)
            iio.imwrite(path, np.asarray(screenshot))
        _run_with_busy_dialog("Export TIFF", "Exporting current view TIFF...", export_tiff)

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
    if_dialog.resize(920, 430)
    if_dialog_layout = QVBoxLayout(if_dialog)
    if_dialog_layout.setContentsMargins(8, 8, 8, 8)
    if_dialog_layout.setSpacing(8)
    if_dialog_layout.addWidget(if_layer_controls)

    def open_if_dialog():
        if_dialog.show()
        if_dialog.raise_()
        if_dialog.activateWindow()

    if_button.clicked.connect(open_if_dialog)

    threshold_launcher = QWidget()
    threshold_launcher_layout = QVBoxLayout(threshold_launcher)
    threshold_launcher_layout.setContentsMargins(0, 0, 0, 0)
    threshold_launcher_layout.setSpacing(6)
    threshold_button = QPushButton("Open MSI Threshold Tools")
    threshold_launcher_layout.addWidget(threshold_button)
    threshold_launcher_layout.addWidget(QLabel("Preview MSI intensity masks and save them as annotations"))

    threshold_dialog = QDialog()
    threshold_dialog.setWindowTitle("MSI Threshold Tools")
    threshold_dialog.setModal(False)
    threshold_dialog.resize(430, 360)
    threshold_dialog_layout = QVBoxLayout(threshold_dialog)
    threshold_dialog_layout.setContentsMargins(8, 8, 8, 8)
    threshold_dialog_layout.setSpacing(8)
    threshold_dialog_layout.addWidget(threshold_preview_controls.native)
    threshold_dialog_layout.addWidget(threshold_percentile_controls.native)
    threshold_dialog_layout.addWidget(threshold_value_label)
    threshold_dialog_layout.addWidget(threshold_selected_count_label)
    threshold_dialog_layout.addWidget(create_msi_threshold_annotation_from_preview.native)
    threshold_dialog_layout.addWidget(delete_threshold_annotation_widget.native)
    def close_threshold_dialog(*_args):
        nonlocal threshold_preview_updates_enabled
        threshold_preview_updates_enabled = False
        threshold_preview_update_timer.stop()
        remove_threshold_preview_layers()

    threshold_dialog.finished.connect(close_threshold_dialog)

    def open_threshold_dialog():
        nonlocal threshold_preview_updates_enabled
        threshold_preview_updates_enabled = False
        state = get_active_state()
        refresh_threshold_prefilter_choices(state)
        threshold_map_to_dataset.value = dataset_choice_text(state)
        threshold_target_mz.value = f"{float(state['current_target_mz']):.4f}"
        threshold_ppm_tolerance.value = float(state["current_ppm_tolerance"])
        threshold_normalize_to_tic.value = bool(state["current_normalize_to_tic"])
        threshold_dialog.show()
        threshold_dialog.raise_()
        threshold_dialog.activateWindow()
        threshold_preview_updates_enabled = True
        _run_with_busy_dialog(
            "MSI Threshold Preview",
            "Updating MSI threshold preview...",
            update_threshold_preview_from_widget,
        )

    threshold_button.clicked.connect(open_threshold_dialog)

    if_threshold_launcher = QWidget()
    if_threshold_launcher_layout = QVBoxLayout(if_threshold_launcher)
    if_threshold_launcher_layout.setContentsMargins(0, 0, 0, 0)
    if_threshold_launcher_layout.setSpacing(6)
    if_threshold_button = QPushButton("Open IF Threshold Tools")
    if_threshold_launcher_layout.addWidget(if_threshold_button)
    if_threshold_launcher_layout.addWidget(QLabel("Preview fluorescence intensity masks and save them as annotations"))

    if_threshold_dialog = QDialog()
    if_threshold_dialog.setWindowTitle("IF Threshold Tools")
    if_threshold_dialog.setModal(False)
    if_threshold_dialog.resize(430, 360)
    if_threshold_dialog_layout = QVBoxLayout(if_threshold_dialog)
    if_threshold_dialog_layout.setContentsMargins(8, 8, 8, 8)
    if_threshold_dialog_layout.setSpacing(8)
    if_threshold_dialog_layout.addWidget(if_threshold_preview_controls.native)
    if_threshold_dialog_layout.addWidget(if_threshold_percentile_controls.native)
    if_threshold_dialog_layout.addWidget(if_threshold_value_label)
    if_threshold_dialog_layout.addWidget(if_threshold_selected_count_label)
    if_threshold_dialog_layout.addWidget(create_if_threshold_annotation_from_preview.native)
    if_threshold_dialog_layout.addWidget(delete_threshold_annotation_widget.native)
    def close_if_threshold_dialog(*_args):
        nonlocal if_threshold_preview_updates_enabled
        if_threshold_preview_updates_enabled = False
        if_threshold_preview_update_timer.stop()
        remove_threshold_preview_layers()

    if_threshold_dialog.finished.connect(close_if_threshold_dialog)

    def open_if_threshold_dialog():
        nonlocal if_threshold_preview_updates_enabled
        if_threshold_preview_updates_enabled = False
        state = get_active_state()
        refresh_if_threshold_choices(state)
        if_threshold_dialog.show()
        if_threshold_dialog.raise_()
        if_threshold_dialog.activateWindow()
        if_threshold_preview_updates_enabled = True
        _run_with_busy_dialog(
            "IF Threshold Preview",
            "Updating fluorescence threshold preview...",
            update_if_threshold_preview_from_widget,
        )

    if_threshold_button.clicked.connect(open_if_threshold_dialog)

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
    alignment_dialog.resize(460, 620)
    alignment_dialog_layout = QVBoxLayout(alignment_dialog)
    alignment_dialog_layout.setContentsMargins(8, 8, 8, 8)
    alignment_dialog_layout.setSpacing(8)
    alignment_dialog_scroll = QScrollArea()
    alignment_dialog_scroll.setWidgetResizable(True)
    alignment_dialog_container = QWidget()
    alignment_dialog_container_layout = QVBoxLayout(alignment_dialog_container)
    alignment_dialog_container_layout.setContentsMargins(4, 4, 4, 4)
    alignment_dialog_container_layout.setSpacing(8)
    alignment_active_dataset_label = QLabel("")
    alignment_dialog_container_layout.addWidget(alignment_active_dataset_label)
    alignment_dialog_container_layout.addWidget(QLabel("Landmark picking"))
    alignment_dialog_container_layout.addWidget(pick_msi_landmarks_widget.native)
    alignment_dialog_container_layout.addWidget(pick_reference_landmarks_widget.native)
    alignment_dialog_container_layout.addWidget(stop_landmark_picking_widget.native)
    alignment_dialog_container_layout.addWidget(QLabel("Alignment actions"))
    alignment_dialog_container_layout.addWidget(fit_affine_from_landmarks.native)
    alignment_dialog_container_layout.addWidget(rotate_180.native)
    alignment_dialog_container_layout.addWidget(rotate_90_cw.native)
    alignment_dialog_container_layout.addWidget(rotate_90_ccw.native)
    alignment_dialog_container_layout.addWidget(flip_horizontal.native)
    alignment_dialog_container_layout.addWidget(flip_vertical.native)
    alignment_dialog_container_layout.addWidget(clear_landmarks.native)
    alignment_dialog_container_layout.addWidget(QLabel("Mutual-information refinement"))
    alignment_dialog_container_layout.addWidget(preview_affine_mi_inputs_widget.native)
    alignment_dialog_container_layout.addWidget(optimize_affine_registration_widget.native)
    alignment_dialog_container_layout.addWidget(QLabel("Registration"))
    alignment_dialog_container_layout.addWidget(save_registration_widget.native)
    alignment_dialog_container_layout.addWidget(save_all_registrations_widget.native)
    alignment_dialog_container_layout.addStretch(1)
    alignment_dialog_scroll.setWidget(alignment_dialog_container)
    alignment_dialog_layout.addWidget(alignment_dialog_scroll)

    def open_alignment_dialog():
        preview_affine_mi_inputs_widget.reference_channel.choices = _reference_channel_choice_names()
        if preview_affine_mi_inputs_widget.reference_channel.value not in preview_affine_mi_inputs_widget.reference_channel.choices:
            preview_affine_mi_inputs_widget.reference_channel.value = preview_affine_mi_inputs_widget.reference_channel.choices[0]
        optimize_affine_registration_widget.reference_channel.choices = _reference_channel_choice_names()
        if optimize_affine_registration_widget.reference_channel.value not in optimize_affine_registration_widget.reference_channel.choices:
            optimize_affine_registration_widget.reference_channel.value = optimize_affine_registration_widget.reference_channel.choices[0]
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
    viewer.window.add_dock_widget(threshold_launcher, area="right", name="MSI Threshold")
    viewer.window.add_dock_widget(if_threshold_launcher, area="right", name="IF Threshold")
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
