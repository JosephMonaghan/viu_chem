from __future__ import annotations

import sys
import tempfile
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
from magicgui import magicgui
from matplotlib import colormaps as mpl_colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from napari.utils.colormaps import Colormap
from qtpy.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QMessageBox,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from shapely.geometry import Point
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


def _clone_spatial_image_element(image) -> Any:
    data = np.asarray(image).copy()
    dims = tuple(getattr(image, "dims", ()))
    if dims and len(dims) == data.ndim:
        return Image2DModel.parse(data, dims=dims)
    return _parse_image_to_spatial(data)


def _to_napari_image(arr: np.ndarray):
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim == 3 and arr.shape[0] in (3, 4):
        return np.moveaxis(arr, 0, -1)
    return arr


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

    def find_local_max_idx_near_mz(self, target_mz: float) -> int:
        local_idx = int(np.argmin(np.abs(self.mz_values[self.local_maxima_indices] - target_mz)))
        return int(self.local_maxima_indices[local_idx])

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
) -> CoregistrationDataset:
    if key not in {"optical", "hne"}:
        raise ValueError("Reference image `key` must be either 'optical' or 'hne'.")

    host_zarr_path = Path(zarr_path).expanduser()
    dataset = CoregistrationDataset(host_zarr_path, registered_cs=registered_cs)
    img = iio.imread(Path(image_path).expanduser())
    element = _parse_image_to_spatial(img)

    px_um_x = 2.54
    px_um_y = 2.54
    try:
        meta = iio.immeta(Path(image_path).expanduser())
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

    element.attrs["pixel_size_x_um"] = float(px_um_x)
    element.attrs["pixel_size_y_um"] = float(px_um_y)
    element.attrs["pixel_size_source"] = "image_metadata_or_default_10000dpi"

    dataset.sdata[key] = element
    dataset.sdata.write_element(key, overwrite=True)
    set_transformation(dataset.sdata.images[key], Identity(), to_coordinate_system="global", write_to_sdata=dataset.sdata)
    set_transformation(dataset.sdata.images[key], Identity(), to_coordinate_system=registered_cs, write_to_sdata=dataset.sdata)
    dataset.sdata.write_transformations(key)
    dataset.sdata.write_consolidated_metadata()
    return dataset


def import_geojson_annotations(
    zarr_path: str | Path,
    annotation_paths: Iterable[str | Path],
    *,
    target_image: str = "hne",
    name_prefix: str = "anno_",
    registered_cs: str = "registered",
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

    imported = []
    for annotation_path in annotation_paths:
        src = Path(annotation_path).expanduser()
        gdf = gpd.read_file(src)
        if gdf.empty:
            continue
        key = f"{name_prefix}{sanitize_name(src.stem)}"
        dataset.sdata[key] = ShapesModel.parse(gdf)
        dataset.sdata.write_element(key, overwrite=True)
        for cs, transform in target_transforms.items():
            set_transformation(
                dataset.sdata.shapes[key],
                transform,
                to_coordinate_system=cs,
                write_to_sdata=dataset.sdata,
            )
        if target_transforms:
            dataset.sdata.write_transformations(key)
        imported.append(key)

    dataset.sdata.write_consolidated_metadata()
    return imported


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
    suppress_registration_control_autocall = False

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
        roi_mask_layer = state.get("roi_mask_layer")
        if roi_mask_layer is not None:
            roi_mask_layer.affine = aff_yx
            roi_mask_layer.scale = (1.0, 1.0)
            roi_mask_layer.translate = (0.0, 0.0)

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

    def add_dataset_to_view(coreg_dataset: CoregistrationDataset, label: str):
        state = build_dataset_state(coreg_dataset, label)
        datasets[str(state["id"])] = state
        apply_transform_to_state(state)
        for idx, key in enumerate(coreg_dataset.reference_image_keys):
            arr = _to_napari_image(np.asarray(coreg_dataset.sdata.images[key]))
            if key in reference_layers:
                reference_layers[key].data = arr
            else:
                reference_layers[key] = viewer.add_image(arr, name=key, visible=(idx == 0 and len(reference_layers) == 0))
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
    annotation_edge_width = 20.0

    def enforce_reference_layers_at_bottom():
        ordered = [key for key in ("optical", "hne") if key in reference_layers]
        ordered.extend([key for key in reference_layers if key not in ordered])
        for target_idx, key in enumerate(ordered):
            layer = reference_layers.get(key)
            if layer is None:
                continue
            try:
                current_idx = viewer.layers.index(layer)
            except Exception:
                continue
            if current_idx != target_idx:
                viewer.layers.move(current_idx, target_idx)

    def add_or_update_reference_layer(source_dataset: CoregistrationDataset, key: str, *, visible: bool = True):
        arr = _to_napari_image(np.asarray(source_dataset.sdata.images[key]))
        if key in reference_layers:
            reference_layers[key].data = arr
            reference_layers[key].visible = visible
        else:
            reference_layers[key] = viewer.add_image(arr, name=key, visible=visible)
        enforce_reference_layers_at_bottom()

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
            except Exception:
                pass

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
            for geom in gdf.geometry:
                for arr_yx, stype in geometry_to_napari_shapes(geom):
                    shape_data.append(arr_yx)
                    shape_types.append(stype)
            if not shape_data:
                continue
            layer_name = f"anno:{state['label']}:{key}"
            if layer_name in annotation_shape_layers:
                annotation_shape_layers[layer_name].data = shape_data
                annotation_shape_layers[layer_name].visible = True
                continue
            annotation_shape_layers[layer_name] = viewer.add_shapes(
                shape_data,
                shape_type=shape_types,
                name=layer_name,
                edge_color=colors[idx % len(colors)],
                face_color=[0, 0, 0, 0],
                edge_width=float(annotation_edge_width),
                opacity=0.9,
                blending="translucent",
                visible=True,
            )
        apply_annotation_visuals()

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
    spectrum_layout.addWidget(QLabel("Average spectrum (click to select m/z)"))
    spectrum_widget.setMinimumHeight(360)
    spectrum_widget.setMinimumWidth(560)
    spectrum_fig = Figure(figsize=(7.5, 3.8), constrained_layout=True)
    spectrum_canvas = FigureCanvas(spectrum_fig)
    spectrum_ax = spectrum_fig.add_subplot(111)
    spectrum_ax.set_xlabel("m/z")
    spectrum_ax.set_ylabel("Average intensity")
    current_mz_line = None
    spectrum_layout.addWidget(spectrum_canvas)

    def redraw_spectrum_for_active_dataset():
        nonlocal current_mz_line
        state = get_active_state()
        coreg_dataset = state["dataset"]
        spectrum_ax.clear()
        spectrum_ax.vlines(coreg_dataset.mz_values, 0, coreg_dataset.avg_spectrum, color="#2c7fb8", linewidth=0.7, alpha=0.9)
        spectrum_ax.set_xlabel("m/z")
        spectrum_ax.set_ylabel("Average intensity")
        spectrum_ax.set_title(f"Average spectrum: {state['label']}")
        current_mz_line = spectrum_ax.axvline(
            coreg_dataset.mz_values[state["current_feature_idx"]],
            color="#d7191c",
            linewidth=1.2,
            alpha=0.9,
        )
        spectrum_canvas.draw_idle()

    def on_spectrum_click(event):
        if event.xdata is None:
            return
        state = get_active_state()
        coreg_dataset = state["dataset"]
        idx = coreg_dataset.find_local_max_idx_near_mz(float(event.xdata))
        mz_selector.target_mz.value = f"{float(coreg_dataset.mz_values[idx]):.4f}"
        update_ion_view_for_mz(float(coreg_dataset.mz_values[idx]), float(state["current_ppm_tolerance"]))

    spectrum_canvas.mpl_connect("button_press_event", on_spectrum_click)

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
        region_mode={"widget_type": "ComboBox", "choices": ["whole", "necrotic_pooled", "regular_only"]},
        auto_call=True,
    )
    def roi_mask_controls(show_mask: bool = False, roi_shape_key: str = "(none)", region_mode: str = "regular_only"):
        state = get_active_state()
        coreg_dataset = state["dataset"]
        if not show_mask or roi_shape_key not in coreg_dataset.sdata.shapes:
            roi_mask_layer = state.get("roi_mask_layer")
            if roi_mask_layer is not None:
                roi_mask_layer.visible = False
            return
        roi_mask_layer = ensure_roi_mask_layer(state)
        rois = coreg_dataset.sdata.transform_element_to_coordinate_system(roi_shape_key, registered_cs)
        areas = np.array([geom.area for geom in rois.geometry], dtype=float)
        whole_idx = int(np.argmax(areas))
        nec_idxs = [idx for idx in range(len(rois)) if idx != whole_idx]
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
        whole = inside[whole_idx] if inside.size else np.zeros(len(points), dtype=bool)
        nec = np.any(inside[nec_idxs], axis=0) if nec_idxs else np.zeros(len(points), dtype=bool)
        selected = whole if region_mode == "whole" else nec if region_mode == "necrotic_pooled" else whole & (~nec)
        mask = np.zeros((coreg_dataset.ny, coreg_dataset.nx), dtype=np.uint8)
        mask[coreg_dataset.y_coords[selected], coreg_dataset.x_coords[selected]] = 1
        roi_mask_layer.data = mask
        roi_mask_layer.visible = True

    @magicgui(edge_width={"widget_type": "FloatSlider", "min": 1.0, "max": 40.0, "step": 0.5}, auto_call=True)
    def annotation_display_controls(edge_width: float = 20.0):
        nonlocal annotation_edge_width
        annotation_edge_width = float(edge_width)
        apply_annotation_visuals()

    @magicgui(
        ty={"widget_type": "FloatSpinBox", "step": 1.0, "min": -1e9, "max": 1e9},
        tx={"widget_type": "FloatSpinBox", "step": 1.0, "min": -1e9, "max": 1e9},
        sy={"widget_type": "FloatSpinBox", "step": 0.01, "min": -1e6, "max": 1e6},
        sx={"widget_type": "FloatSpinBox", "step": 0.01, "min": -1e6, "max": 1e6},
        auto_call=True,
    )
    def registration_controls(
        ty: float = float(initial_state["current_transform_xy"][1, 2]),
        tx: float = float(initial_state["current_transform_xy"][0, 2]),
        sy: float = float(initial_state["current_transform_xy"][1, 1]),
        sx: float = float(initial_state["current_transform_xy"][0, 0]),
    ):
        nonlocal suppress_registration_control_autocall
        if suppress_registration_control_autocall:
            return
        state = get_active_state()
        updated = np.asarray(state["current_transform_xy"], dtype=float).copy()
        updated[0, 0] = float(sx)
        updated[1, 1] = float(sy)
        updated[0, 2] = float(tx)
        updated[1, 2] = float(ty)
        state["current_transform_xy"][:] = updated
        apply_transform_to_state(state)
        try:
            roi_mask_controls()
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

    def sync_controls_to_active_dataset():
        nonlocal suppress_registration_control_autocall
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
        rebuild_msi_layer_controls()
        suppress_registration_control_autocall = True
        try:
            registration_controls.ty.value = float(state["current_transform_xy"][1, 2])
            registration_controls.tx.value = float(state["current_transform_xy"][0, 2])
            registration_controls.sy.value = float(state["current_transform_xy"][1, 1])
            registration_controls.sx.value = float(state["current_transform_xy"][0, 0])
        finally:
            suppress_registration_control_autocall = False
        for key, other_state in datasets.items():
            other_state["msi_landmarks"].visible = (key == str(state["id"]))
            if key != str(state["id"]):
                roi_mask_layer = other_state.get("roi_mask_layer")
                if roi_mask_layer is not None:
                    roi_mask_layer.visible = False
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

    @magicgui(target_image={"widget_type": "LineEdit"}, name_prefix={"widget_type": "LineEdit"}, call_button="Add GeoJSON")
    def add_geojson_annotations(target_image: str = "hne", name_prefix: str = "anno_"):
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
        )
        coreg_dataset.sdata = sd.read_zarr(coreg_dataset.zarr_path)
        add_annotation_shape_layers(state, keys)
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

    controls_scroll = QScrollArea()
    controls_scroll.setWidgetResizable(True)
    controls_container = QWidget()
    controls_layout = QVBoxLayout(controls_container)
    controls_layout.setContentsMargins(6, 6, 6, 6)
    controls_layout.setSpacing(8)
    controls_layout.addWidget(spectrum_widget)
    controls_layout.addWidget(rename_dataset_widget.native)
    controls_layout.addWidget(copy_affine_to_target_widget.native)
    controls_layout.addWidget(msi_layer_controls)
    controls_layout.addWidget(ion_display_options.native)
    controls_layout.addWidget(mz_selector.native)
    controls_layout.addWidget(roi_mask_controls.native)
    controls_layout.addWidget(annotation_display_controls.native)
    controls_layout.addWidget(registration_controls.native)
    controls_layout.addWidget(fit_affine_from_landmarks.native)
    controls_layout.addWidget(rotate_180.native)
    controls_layout.addWidget(rotate_90_cw.native)
    controls_layout.addWidget(rotate_90_ccw.native)
    controls_layout.addWidget(flip_horizontal.native)
    controls_layout.addWidget(flip_vertical.native)
    controls_layout.addWidget(clear_landmarks.native)
    controls_layout.addWidget(save_registration_widget.native)
    controls_layout.addWidget(save_all_registrations_widget.native)
    controls_layout.addStretch(1)
    controls_scroll.setWidget(controls_container)

    viewer.window.add_dock_widget(controls_scroll, area="right", name="Controls")
    viewer.window.add_dock_widget(add_msi_dataset, area="left")
    viewer.window.add_dock_widget(add_optical_image, area="left")
    viewer.window.add_dock_widget(add_hne_image, area="left")
    viewer.window.add_dock_widget(add_geojson_annotations, area="left")
    enforce_reference_layers_at_bottom()
    add_annotation_shape_layers(initial_state)
    sync_controls_to_active_dataset()
    try:
        viewer.reset_view()
    except Exception:
        pass
    napari.run()
    return viewer
