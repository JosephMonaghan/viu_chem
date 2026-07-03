from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import imageio.v3 as iio
import napari
import numpy as np
import pandas as pd
import spatialdata as sd
import zarr
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
from shapely.geometry import MultiPolygon, Point, Polygon
from spatialdata.models import ShapesModel
from spatialdata.transformations import get_transformation, set_transformation

from . import msi_coregistration as _api
from .coreg_figures import _msi_pixel_size_um, _reference_display_pixel_size_from_msi
from .msi_coregistration import (
    CoregistrationDataset,
    REFERENCE_CHANNEL_COLOR_PRESETS,
    _annotation_mask_from_transformed_geometries,
    _fallback_reference_channel_color,
    _infer_msi_dataset_specs,
    _make_reference_channel_colormap,
    _sample_msi_values_at_msi_pixels,
    _sample_reference_values_at_msi_pixels,
    _sanitize_dataset_label,
    _to_napari_image,
    _write_element_to_existing_store,
    add_reference_image,
    auto_contrast_limits,
    convert_input_to_zarr,
    create_msi_threshold_annotation,
    create_pooled_msi_threshold_annotation,
    create_reference_threshold_annotation,
    delete_geojson_annotations,
    delete_msi_dataset,
    embed_msi_dataset,
    finite_data_limits,
    import_geojson_annotations,
    normalize_image_for_registration,
    prepare_coregistration_zarr,
    prepare_ion_for_display,
    rename_msi_dataset,
    sanitize_name,
    save_coregistration,
    sitk_affine_from_fixed_to_moving_matrix,
    sitk_transform_to_homogeneous_matrix,
    transform_geojson_annotations,
    xy_to_yx_matrix,
)

globals().update({name: getattr(_api, name) for name in dir(_api) if not name.startswith("__")})

_QT_APP = None


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

    viewer = napari.Viewer(title=host_zarr_path.name)
    retained_dialogs: list[QDialog] = []

    def retain_dialog(dialog: QDialog):
        retained_dialogs.append(dialog)

        def forget_dialog(*_args, dialog=dialog):
            try:
                retained_dialogs.remove(dialog)
            except ValueError:
                pass

        try:
            dialog.destroyed.connect(forget_dialog)
        except Exception:
            pass
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

    ion_view_state_keys = (
        "current_feature_idx",
        "current_target_mz",
        "current_ppm_tolerance",
        "current_feature_indices",
        "current_normalize_to_tic",
        "current_normalization_mode",
        "current_ratio_mz",
        "current_ratio_feature_indices",
        "current_ratio_actual_mzs",
        "current_ratio_status",
        "current_colormap_name",
        "current_opacity",
        "current_contrast_mode",
        "current_contrast_low_pct",
        "current_contrast_high_pct",
        "current_contrast_low",
        "current_contrast_high",
    )

    def copy_ion_view_state(state: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in ion_view_state_keys:
            value = state.get(key)
            out[key] = value.copy() if isinstance(value, np.ndarray) else value
        return out

    def load_ion_view_state(state: dict[str, Any], viewer_record: dict[str, Any]):
        for key in ion_view_state_keys:
            if key not in viewer_record:
                continue
            value = viewer_record[key]
            state[key] = value.copy() if isinstance(value, np.ndarray) else value
        state["ion_layer"] = viewer_record["layer"]

    def sync_active_ion_viewer_record(state: dict[str, Any]):
        viewers = state.get("ion_viewers", [])
        if not viewers:
            return
        idx = int(state.get("active_ion_viewer_index", 0))
        if idx < 0 or idx >= len(viewers):
            idx = 0
            state["active_ion_viewer_index"] = 0
        viewers[idx].update(copy_ion_view_state(state))
        viewers[idx]["layer"] = state["ion_layer"]

    def ion_viewer_choice_text(state: dict[str, Any], idx: int) -> str:
        viewers = state.get("ion_viewers", [])
        if idx < 0 or idx >= len(viewers):
            return "Viewer 1"
        return str(viewers[idx].get("name") or f"Viewer {idx + 1}")

    def ion_viewer_choices(state: dict[str, Any]) -> list[str]:
        return [ion_viewer_choice_text(state, idx) for idx in range(len(state.get("ion_viewers", [])))] or ["Viewer 1"]

    def ion_viewer_index_from_choice(state: dict[str, Any], choice: str) -> int:
        text = str(choice)
        for idx, label in enumerate(ion_viewer_choices(state)):
            if text == label:
                return idx
        return int(state.get("active_ion_viewer_index", 0))

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
        initial_transform_xy = np.asarray(saved_xy_matrix, dtype=float).copy()

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
            "current_normalization_mode": "tic",
            "current_ratio_mz": "",
            "current_ratio_feature_indices": np.array([], dtype=int),
            "current_ratio_actual_mzs": np.array([], dtype=float),
            "current_ratio_status": "",
            "current_colormap_name": overlay_name,
            "current_opacity": 0.6,
            "current_contrast_mode": "percentile",
            "current_contrast_low_pct": 1.0,
            "current_contrast_high_pct": 99.5,
            "current_contrast_low": float(initial_contrast_limits[0]),
            "current_contrast_high": float(initial_contrast_limits[1]),
            "current_transform_xy": initial_transform_xy.copy(),
            "initial_transform_xy": initial_transform_xy.copy(),
            "current_transform_is_initial_guess": not bool(found_registration),
            "ion_layer": ion_layer,
            "active_ion_viewer_index": 0,
            "ion_viewers": [],
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
        state["ion_viewers"] = [{"name": "Viewer 1", "layer": ion_layer, **copy_ion_view_state(state)}]
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
        for viewer_record in list(state.get("ion_viewers", [])):
            layer = viewer_record.get("layer")
            try:
                layer.affine = aff_yx
                layer.scale = (1.0, 1.0)
                layer.translate = (0.0, 0.0)
            except Exception:
                pass
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
        try:
            schedule_scale_bar_update()
        except NameError:
            pass

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

    def refresh_datasets_after_reference_update():
        refreshed_sdata = sd.read_zarr(host_zarr_path)
        specs = _infer_msi_dataset_specs(refreshed_sdata)
        all_tic_keys = {str(spec["tic_key"]) for spec in specs}
        reference_keys = [key for key in refreshed_sdata.images.keys() if key not in all_tic_keys]
        for state in datasets.values():
            coreg_dataset = state["dataset"]
            coreg_dataset.sdata = refreshed_sdata
            coreg_dataset.reference_image_keys = list(reference_keys)
            current_transform = np.asarray(state["current_transform_xy"], dtype=float)
            initial_transform = np.asarray(state.get("initial_transform_xy", np.eye(3, dtype=float)), dtype=float)
            if not bool(state.get("current_transform_is_initial_guess", False)):
                continue
            if current_transform.shape != (3, 3) or not np.allclose(current_transform, initial_transform, rtol=1e-6, atol=1e-6):
                continue
            scale_guess, info = coreg_dataset.estimate_initial_scale_from_pixel_sizes()
            if info is None:
                continue
            scale_guess = np.asarray(scale_guess, dtype=float)
            if scale_guess.shape != (3, 3) or not np.all(np.isfinite(scale_guess)):
                continue
            if np.allclose(scale_guess, current_transform, rtol=1e-6, atol=1e-6):
                continue
            state["current_transform_xy"][:] = scale_guess
            state["initial_transform_xy"] = scale_guess.copy()
            apply_transform_to_state(state)

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
                if "pixels" in key.lower():
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
        inclusion_mode: str = "center",
        min_hole_area: float = 0.0,
    ) -> np.ndarray:
        coreg_dataset = state["dataset"]
        if roi_shape_key not in coreg_dataset.sdata.shapes:
            return np.zeros(len(coreg_dataset.x_coords), dtype=bool)
        rois = coreg_dataset.sdata.transform_element_to_coordinate_system(roi_shape_key, registered_cs)
        if region_label == "(all regions)" or "_annotation_label" not in rois.columns:
            selected_rois = rois
        else:
            matching_idxs = [
                idx for idx, value in enumerate(rois["_annotation_label"])
                if str(value).strip() == str(region_label).strip()
            ]
            selected_rois = rois.iloc[matching_idxs].copy() if matching_idxs else rois.iloc[[]].copy()
        return _annotation_mask_from_transformed_geometries(
            coreg_dataset,
            selected_rois,
            np.asarray(state["current_transform_xy"], dtype=float),
            inclusion_mode=inclusion_mode,
            min_hole_area=min_hole_area,
        )

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

    def selected_annotation_rows(active_layer, *, include_same_label: bool = False) -> tuple[dict[str, Any] | None, str | None, list[int], str]:
        if active_layer is None or active_layer not in annotation_shape_layers.values():
            return None, None, [], ""
        selected_data = sorted(int(idx) for idx in getattr(active_layer, "selected_data", set()))
        if not selected_data:
            return None, None, [], ""
        metadata = _layer_metadata(active_layer)
        dataset_id = str(metadata.get("annotation_dataset_id", ""))
        shape_key = str(metadata.get("annotation_shape_key", ""))
        row_lookup = list(metadata.get("annotation_source_row_indices", []))
        if not dataset_id or not shape_key or not row_lookup:
            return None, None, [], ""
        source_state = datasets.get(dataset_id)
        if source_state is None or shape_key not in source_state["dataset"].sdata.shapes:
            return None, None, [], ""

        source_gdf = source_state["dataset"].sdata.shapes[shape_key]
        row_indices = sorted({int(row_lookup[idx]) for idx in selected_data if 0 <= int(idx) < len(row_lookup)})
        current_label = ""
        if "_annotation_label" in source_gdf.columns:
            labels = [
                str(source_gdf.iloc[idx]["_annotation_label"]).strip()
                for idx in row_indices
                if 0 <= idx < len(source_gdf) and str(source_gdf.iloc[idx]["_annotation_label"]).strip()
            ]
            current_label = labels[0] if labels else ""
            if include_same_label and current_label:
                row_indices = [
                    idx for idx, value in enumerate(source_gdf["_annotation_label"])
                    if str(value).strip() == current_label
                ]
        return source_state, shape_key, row_indices, current_label

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
        if "remove_annotation_from_zarr_widget" in locals():
            remove_annotation_from_zarr_widget.annotation_key.choices = choices
            if remove_annotation_from_zarr_widget.annotation_key.value not in choices:
                remove_annotation_from_zarr_widget.annotation_key.value = choices[0]

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

    def remove_annotation_shape_layers_for_dataset(dataset_id: str):
        remove_dataset_id = str(dataset_id)
        for layer_name in list(annotation_shape_layers.keys()):
            layer = annotation_shape_layers.get(layer_name)
            if layer is None:
                continue
            metadata = _layer_metadata(layer)
            if str(metadata.get("annotation_dataset_id", "")) != remove_dataset_id:
                continue
            annotation_shape_layers.pop(layer_name, None)
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

    scale_bar_settings: dict[str, Any] = {
        "visible": True,
        "location": "lower right",
        "color": "white",
        "font_size": 12,
    }
    scale_bar_update_timer = QTimer()
    scale_bar_update_timer.setSingleShot(True)

    def _active_reference_key_for_scale_bar() -> str | None:
        visible_layers = [layer for layer in _reference_channel_layers() if bool(getattr(layer, "visible", False))]
        for layer in visible_layers + _reference_channel_layers():
            metadata = _layer_metadata(layer)
            reference_key = str(metadata.get("reference_key", "")).strip()
            if reference_key:
                return reference_key
        state = get_active_state()
        keys = list(state["dataset"].reference_image_keys)
        if "hne" in keys:
            return "hne"
        if "optical" in keys:
            return "optical"
        return keys[0] if keys else None

    def _active_reference_display_pixel_size_um(state: dict[str, Any]) -> tuple[float, float] | None:
        msi_pixel_size = _msi_pixel_size_um(state["dataset"])
        transform_xy = np.asarray(state.get("current_transform_xy"), dtype=float)
        if msi_pixel_size is not None and transform_xy.shape == (3, 3) and np.all(np.isfinite(transform_xy)):
            origin = transform_xy @ np.array([0.0, 0.0, 1.0])
            x_step = transform_xy @ np.array([1.0, 0.0, 1.0])
            y_step = transform_xy @ np.array([0.0, 1.0, 1.0])
            x_pixels = float(np.linalg.norm(x_step[:2] - origin[:2]))
            y_pixels = float(np.linalg.norm(y_step[:2] - origin[:2]))
            if np.isfinite(x_pixels) and x_pixels > 0 and np.isfinite(y_pixels) and y_pixels > 0:
                return float(msi_pixel_size[0]) / x_pixels, float(msi_pixel_size[1]) / y_pixels

        reference_key = _active_reference_key_for_scale_bar()
        if reference_key is None:
            return None
        return _reference_display_pixel_size_from_msi(state["dataset"], reference_key)

    def update_scale_bar_overlay():
        state = get_active_state()
        pixel_size_um = _active_reference_display_pixel_size_um(state)
        scale_bar = viewer.scale_bar
        scale_bar.visible = bool(scale_bar_settings.get("visible", True)) and pixel_size_um is not None
        if pixel_size_um is None:
            return

        position_map = {
            "lower right": "bottom_right",
            "lower left": "bottom_left",
            "upper right": "top_right",
            "upper left": "top_left",
        }
        location = str(scale_bar_settings.get("location", "lower right")).lower().replace("_", " ")
        scale_bar.position = position_map.get(location, "bottom_right")
        scale_bar.unit = f"{float(pixel_size_um[0]):.12g} um"
        scale_bar.length = None
        scale_bar.colored = True
        scale_bar.color = str(scale_bar_settings.get("color", "white"))
        scale_bar.box = True
        scale_bar.box_color = [0.0, 0.0, 0.0, 0.55]
        scale_bar.ticks = True
        scale_bar.font_size = int(scale_bar_settings.get("font_size", 12))

    def schedule_scale_bar_update(*_args):
        scale_bar_update_timer.start(40)

    scale_bar_update_timer.timeout.connect(update_scale_bar_overlay)

    def normalize_ion_display_mode(mode: str) -> str:
        value = getattr(mode, "value", mode)
        text = str(value).strip().lower()
        aliases = {
            "tic": "tic",
            "TIC": "tic",
            "normalize to tic": "tic",
            "tic-normalized": "tic",
            "none": "raw",
            "raw": "raw",
            "raw intensity": "raw",
            "m/z ratio": "mz_ratio",
            "mz_ratio": "mz_ratio",
            "mz ratio": "mz_ratio",
            "ratio": "mz_ratio",
            "normalize to m/z": "mz_ratio",
        }
        return aliases.get(text, "tic")

    def display_label_for_ion_state(state: dict[str, Any]) -> str:
        target_mz = float(state["current_target_mz"])
        ppm_tolerance = float(state["current_ppm_tolerance"])
        feature_count = len(state.get("current_feature_indices", []))
        if feature_count == 0:
            base = f"{state['label']} m/z {target_mz:.4f} +/- {ppm_tolerance:.1f} ppm (no features)"
        elif feature_count > 1:
            base = f"{state['label']} m/z {target_mz:.4f} +/- {ppm_tolerance:.1f} ppm"
        else:
            base = f"{state['label']} m/z {target_mz:.4f}"
        if str(state.get("current_normalization_mode", "tic")) == "mz_ratio":
            ratio_mz = str(state.get("current_ratio_mz", "")).strip()
            if ratio_mz:
                base = f"{base} / m/z {ratio_mz}"
        return base

    def display_label_for_active_ion_viewer(state: dict[str, Any]) -> str:
        viewer_name = ion_viewer_choice_text(state, int(state.get("active_ion_viewer_index", 0)))
        return f"{viewer_name}: {display_label_for_ion_state(state)}"

    def summarize_ion_display_status(state: dict[str, Any], img: np.ndarray) -> str:
        mode = normalize_ion_display_mode(state.get("current_normalization_mode", "tic"))
        numerator_count = len(state.get("current_feature_indices", []))
        data = np.asarray(img, dtype=float)
        finite = data[np.isfinite(data)]
        positive = finite[finite > 0]
        if positive.size:
            value_text = (
                f"positive n={positive.size}, "
                f"p1={np.percentile(positive, 1):.4g}, "
                f"p99.5={np.percentile(positive, 99.5):.4g}, "
                f"max={np.max(positive):.4g}"
            )
        else:
            value_text = "no positive display pixels"
        if mode != "mz_ratio":
            return f"Normalization: {mode}; numerator features={numerator_count}; {value_text}"

        ratio_text = str(state.get("current_ratio_mz", "")).strip()
        denominator_indices = np.asarray(state.get("current_ratio_feature_indices", []), dtype=int)
        actual_mzs = np.asarray(state.get("current_ratio_actual_mzs", []), dtype=float)
        if not ratio_text:
            return f"Normalization: m/z ratio; enter denominator m/z; numerator features={numerator_count}; {value_text}"
        if denominator_indices.size == 0:
            return (
                f"Normalization: m/z ratio; denominator {ratio_text} has no features "
                f"within +/- {float(state['current_ppm_tolerance']):.1f} ppm; {value_text}"
            )
        actual_text = ", ".join(f"{mz:.4f}" for mz in actual_mzs[:4])
        if actual_mzs.size > 4:
            actual_text = f"{actual_text}, ..."
        return (
            f"Normalization: m/z ratio; numerator features={numerator_count}; "
            f"denominator features={denominator_indices.size} ({actual_text}); {value_text}"
        )

    def update_ion_status_label(state: dict[str, Any], img: np.ndarray):
        status = summarize_ion_display_status(state, img)
        state["current_ratio_status"] = status
        try:
            ion_status_label.setText(status)
        except Exception:
            pass

    def reconstruct_ion_display_image(state: dict[str, Any]) -> np.ndarray:
        coreg_dataset = state["dataset"]
        mode = normalize_ion_display_mode(state.get("current_normalization_mode", "tic"))
        state["current_normalization_mode"] = mode
        state["current_normalize_to_tic"] = mode == "tic"
        numerator = coreg_dataset.reconstruct_ion_image(
            state["current_feature_indices"],
            normalize_to_tic=(mode == "tic"),
        )
        state["current_ratio_feature_indices"] = np.array([], dtype=int)
        state["current_ratio_actual_mzs"] = np.array([], dtype=float)
        if mode != "mz_ratio":
            return numerator

        try:
            ratio_mz = float(str(state.get("current_ratio_mz", "")).strip())
        except Exception:
            return np.zeros_like(numerator, dtype=float)
        denominator_indices = coreg_dataset.find_feature_indices_from_mz(ratio_mz, float(state["current_ppm_tolerance"]))
        state["current_ratio_feature_indices"] = np.asarray(denominator_indices, dtype=int)
        state["current_ratio_actual_mzs"] = coreg_dataset.mz_values[denominator_indices].copy() if denominator_indices.size else np.array([], dtype=float)
        if denominator_indices.size == 0:
            return np.zeros_like(numerator, dtype=float)
        denominator = coreg_dataset.reconstruct_ion_image(denominator_indices, normalize_to_tic=False)
        raw_numerator = coreg_dataset.reconstruct_ion_image(state["current_feature_indices"], normalize_to_tic=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.divide(
                raw_numerator,
                denominator,
                out=np.zeros_like(raw_numerator, dtype=float),
                where=np.isfinite(denominator) & (denominator > 0),
            )

    def update_ion_view(feature_idx: int):
        state = get_active_state()
        coreg_dataset = state["dataset"]
        state["current_feature_idx"] = int(np.clip(feature_idx, 0, len(coreg_dataset.mz_values) - 1))
        state["current_feature_indices"] = np.array([state["current_feature_idx"]], dtype=int)
        state["current_target_mz"] = float(coreg_dataset.mz_values[state["current_feature_idx"]])
        img = reconstruct_ion_display_image(state)
        display_img = prepare_ion_for_display(img)
        state["ion_layer"].data = display_img
        state["ion_layer"].name = display_label_for_active_ion_viewer(state)
        apply_ion_contrast_to_active_layer(display_img)
        update_ion_status_label(state, display_img)
        sync_active_ion_viewer_record(state)
        if current_mz_line is not None:
            current_mz_line.set_xdata(
                [coreg_dataset.mz_values[state["current_feature_idx"]], coreg_dataset.mz_values[state["current_feature_idx"]]]
            )
            spectrum_canvas.draw_idle()

    def update_ion_view_for_mz(target_mz: float, ppm_tolerance: float):
        state = get_active_state()
        coreg_dataset = state["dataset"]
        indices = coreg_dataset.find_feature_indices_from_mz(float(target_mz), float(ppm_tolerance))
        if indices.size:
            nearest_local = int(np.argmin(np.abs(coreg_dataset.mz_values[indices] - float(target_mz))))
            state["current_feature_idx"] = int(indices[nearest_local])
        state["current_feature_indices"] = np.asarray(indices, dtype=int)
        state["current_target_mz"] = float(target_mz)
        state["current_ppm_tolerance"] = float(ppm_tolerance)
        img = reconstruct_ion_display_image(state)
        display_img = prepare_ion_for_display(img)
        state["ion_layer"].data = display_img
        state["ion_layer"].name = display_label_for_active_ion_viewer(state)
        apply_ion_contrast_to_active_layer(display_img)
        update_ion_status_label(state, display_img)
        sync_active_ion_viewer_record(state)
        if current_mz_line is not None:
            current_mz_line.set_xdata([float(target_mz), float(target_mz)])
            spectrum_canvas.draw_idle()

    ion_viewer_widget_syncing = False

    @magicgui(
        viewer={"widget_type": "ComboBox", "choices": ["Viewer 1"]},
        auto_call=True,
    )
    def ion_viewer_selector(viewer: str = "Viewer 1"):
        nonlocal ion_viewer_widget_syncing
        if ion_viewer_widget_syncing:
            return
        state = get_active_state()
        selected_idx = ion_viewer_index_from_choice(state, str(viewer))
        if selected_idx == int(state.get("active_ion_viewer_index", 0)):
            return
        sync_active_ion_viewer_record(state)
        state["active_ion_viewer_index"] = int(selected_idx)
        load_ion_view_state(state, state["ion_viewers"][selected_idx])
        sync_controls_to_active_dataset()

    @magicgui(call_button="Add Ion Viewer")
    def add_ion_viewer_widget():
        state = get_active_state()
        sync_active_ion_viewer_record(state)
        source_layer = state["ion_layer"]
        next_idx = len(state.get("ion_viewers", [])) + 1
        layer = viewer.add_image(
            np.asarray(source_layer.data).copy(),
            name=f"{state['label']} Viewer {next_idx} m/z {float(state.get('current_target_mz', 0.0)):.4f}",
            opacity=float(getattr(source_layer, "opacity", state.get("current_opacity", 0.6))),
            colormap=getattr(source_layer, "colormap", get_overlay_colormap(str(state.get("current_colormap_name", "viridis")))),
            contrast_limits=tuple(float(v) for v in getattr(source_layer, "contrast_limits", (0.0, 1.0))),
            blending="translucent",
            interpolation2d="nearest",
            visible=True,
        )
        record = {"name": f"Viewer {next_idx}", "layer": layer, **copy_ion_view_state(state)}
        state.setdefault("ion_viewers", []).append(record)
        state["active_ion_viewer_index"] = next_idx - 1
        load_ion_view_state(state, record)
        apply_transform_to_state(state)
        sync_controls_to_active_dataset()
        try:
            viewer.layers.selection.active = layer
        except Exception:
            pass

    @magicgui(call_button="Remove Ion Viewer")
    def remove_ion_viewer_widget():
        state = get_active_state()
        viewers = state.get("ion_viewers", [])
        if len(viewers) <= 1:
            QMessageBox.warning(None, "Remove Ion Viewer", "At least one ion viewer must remain for the active dataset.")
            return
        idx = int(state.get("active_ion_viewer_index", 0))
        idx = min(max(idx, 0), len(viewers) - 1)
        removed = viewers.pop(idx)
        _remove_viewer_layer(removed.get("layer"))
        new_idx = min(idx, len(viewers) - 1)
        state["active_ion_viewer_index"] = int(new_idx)
        load_ion_view_state(state, viewers[new_idx])
        sync_controls_to_active_dataset()

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

    ion_status_label = QLabel("")
    ion_status_label.setWordWrap(True)
    ion_status_label.setStyleSheet("color: #ffffff;")

    @magicgui(
        normalization_mode={
            "widget_type": "ComboBox",
            "choices": [("TIC", "tic"), ("raw", "raw"), ("m/z ratio", "mz_ratio")],
            "label": "Ion normalization",
        },
        ratio_mz={
            "widget_type": "LineEdit",
            "label": "Denominator m/z",
        },
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
        normalization_mode: str = "tic",
        ratio_mz: str = "",
        contrast_mode: str = "percentile",
        contrast_percentiles: tuple[float, float] = (1.0, 99.5),
        absolute_low: str = "0.0",
        absolute_high: str = "1.0",
    ):
        state = get_active_state()
        norm_mode = normalize_ion_display_mode(normalization_mode)
        state["current_normalization_mode"] = norm_mode
        state["current_normalize_to_tic"] = norm_mode == "tic"
        state["current_ratio_mz"] = str(ratio_mz).strip()
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
        min_hole_area={"widget_type": "FloatSpinBox", "min": 0.0, "max": 1000000.0, "step": 100.0, "label": "Fill holes smaller than area"},
        auto_call=True,
    )
    def roi_mask_controls(
        show_mask: bool = False,
        roi_shape_key: str = "(none)",
        region_label: str = "(all regions)",
        min_hole_area: float = 0.0,
    ):
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
        selected = compute_annotation_region_mask(state, roi_shape_key, region_label, min_hole_area=float(min_hole_area))
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
        new_label={"widget_type": "LineEdit", "label": "New ROI label"},
        include_same_label={"widget_type": "CheckBox", "text": "Rename all ROIs with the same current label"},
        call_button="Rename Selected ROI(s)",
    )
    def rename_selected_annotation_widget(new_label: str = "", include_same_label: bool = False):
        active_layer = getattr(viewer.layers.selection, "active", None)
        source_state, shape_key, row_indices, current_label = selected_annotation_rows(
            active_layer,
            include_same_label=bool(include_same_label),
        )
        if source_state is None or shape_key is None or not row_indices:
            QMessageBox.warning(
                None,
                "Rename Selected ROI(s)",
                "Select an annotation shapes layer and click one or more shapes first.",
            )
            return
        cleaned_label = str(new_label).strip()
        if not cleaned_label:
            QMessageBox.warning(None, "Rename Selected ROI(s)", "Enter a non-empty ROI label.")
            return

        def rename_rows():
            coreg_dataset = source_state["dataset"]
            source_gdf = coreg_dataset.sdata.shapes[shape_key].copy()
            if "_annotation_label" not in source_gdf.columns:
                source_gdf["_annotation_label"] = ""
            valid_rows = [idx for idx in row_indices if 0 <= int(idx) < len(source_gdf)]
            if not valid_rows:
                raise ValueError("The selected annotation rows are no longer available.")
            source_gdf.loc[source_gdf.index[valid_rows], "_annotation_label"] = cleaned_label
            transforms = get_transformation(coreg_dataset.sdata.shapes[shape_key], get_all=True)
            shape_element = ShapesModel.parse(source_gdf)
            set_transformation(shape_element, transforms, set_all=True)
            _write_element_to_existing_store(
                coreg_dataset.zarr_path,
                element=shape_element,
                element_type="shapes",
                element_name=shape_key,
                overwrite=True,
                consolidate_metadata=True,
            )
            return len(valid_rows)

        try:
            renamed_count = _run_with_busy_dialog(
                "Rename Selected ROI(s)",
                "Updating annotation labels...",
                rename_rows,
            )
        except Exception as exc:
            QMessageBox.warning(None, "Rename Selected ROI(s)", str(exc))
            return

        for dataset_state in datasets.values():
            dataset_state["dataset"].sdata = sd.read_zarr(dataset_state["dataset"].zarr_path)
            add_annotation_shape_layers(dataset_state, [shape_key])
        apply_annotation_visuals()
        refresh_annotation_widget_choices()
        try:
            refresh_threshold_prefilter_choices(get_active_state())
            refresh_if_threshold_choices(get_active_state())
            roi_mask_controls()
        except Exception:
            pass
        old_label = f" from '{current_label}'" if current_label else ""
        QMessageBox.information(
            None,
            "Rename Selected ROI(s)",
            f"Renamed {renamed_count} ROI geometry/geometries{old_label} to '{cleaned_label}'.",
        )

    @magicgui(
        normalize_to_tic={"widget_type": "CheckBox", "text": "Normalize to TIC"},
        call_button="Export MSI From ROI",
    )
    def export_msi_from_roi_widget(normalize_to_tic: bool = True):
        state = get_active_state()
        coreg_dataset = state["dataset"]
        roi_shape_key = str(roi_mask_controls.roi_shape_key.value)
        region_label = str(roi_mask_controls.region_label.value)
        min_hole_area = float(roi_mask_controls.min_hole_area.value)
        if roi_shape_key == "(none)" or roi_shape_key not in coreg_dataset.sdata.shapes:
            QMessageBox.warning(None, "Export MSI From ROI", "Select an annotation layer in ROI selection first.")
            return
        selected = compute_annotation_region_mask(state, roi_shape_key, region_label, min_hole_area=min_hole_area)
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
        return reconstruct_ion_display_image(state)

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
        retain_dialog(dialog)

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
        retain_dialog(dialog)

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
        for idx, viewer_record in enumerate(state.get("ion_viewers", [])):
            viewer_record["name"] = viewer_record.get("name") or f"Viewer {idx + 1}"
            layer = viewer_record.get("layer")
            if layer is not None:
                try:
                    layer.name = f"{viewer_record['name']}: {new_name} m/z {float(viewer_record.get('current_target_mz', state['current_target_mz'])):.4f}"
                except Exception:
                    pass
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
                sync_active_ion_viewer_record(datasets[dataset_key])

            def _set_active(checked, dataset_key=dataset_key):
                if checked:
                    set_active_dataset(dataset_key)

            def _set_colormap(value, dataset_key=dataset_key):
                if dataset_key not in datasets or value not in mpl_colormaps:
                    return
                state = datasets[dataset_key]
                state["current_colormap_name"] = str(value)
                state["ion_layer"].colormap = get_overlay_colormap(str(value))
                sync_active_ion_viewer_record(state)

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

    def _confirm_data_delete(title: str, message: str) -> bool:
        result = QMessageBox.question(
            None,
            title,
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return result == QMessageBox.Yes

    def _remove_viewer_layer(layer):
        if layer is None:
            return
        try:
            viewer.layers.remove(layer)
        except Exception:
            pass

    @magicgui(
        dataset={"widget_type": "ComboBox", "choices": current_dataset_choices()},
        call_button="Remove MSI Dataset From Zarr",
    )
    def remove_msi_dataset_from_zarr_widget(dataset: str = initial_active_choice):
        nonlocal active_dataset_label
        dataset_key = dataset_key_from_choice(str(dataset))
        if dataset_key is None or dataset_key not in datasets:
            return
        if len(datasets) <= 1:
            QMessageBox.warning(None, "Remove MSI Dataset", "At least one MSI dataset must remain open in the viewer.")
            return
        state = datasets[dataset_key]
        if not _confirm_data_delete(
            "Remove MSI Dataset",
            (
                f"Remove MSI dataset '{state['label']}' from the zarr?\n\n"
                "This deletes its table, TIC image, and MSI pixel shapes. Annotation shapes are not deleted."
            ),
        ):
            return

        deleted = _run_with_busy_dialog(
            "Remove MSI Dataset",
            "Deleting MSI dataset from zarr...",
            lambda: delete_msi_dataset(state["dataset"].zarr_path, table_key=state["dataset"].table_key),
        )
        if not any(deleted.values()):
            QMessageBox.warning(None, "Remove MSI Dataset", "No matching MSI dataset components were deleted.")
            return

        remove_annotation_shape_layers_for_dataset(dataset_key)
        seen_layers = set()
        for viewer_record in list(state.get("ion_viewers", [])):
            layer = viewer_record.get("layer")
            if layer is None or id(layer) in seen_layers:
                continue
            seen_layers.add(id(layer))
            _remove_viewer_layer(layer)
        for layer_key in (
            "msi_landmarks",
            "roi_mask_layer",
            "selected_annotation_mask_layer",
            "threshold_preview_layer",
            "optimization_preview_layer",
        ):
            _remove_viewer_layer(state.get(layer_key))
        datasets.pop(dataset_key, None)
        for other_state in datasets.values():
            other_state["dataset"].sdata = sd.read_zarr(other_state["dataset"].zarr_path)
        if str(active_dataset_label) == str(dataset_key):
            active_dataset_label = str(next(iter(datasets.keys())))
            datasets[active_dataset_label]["ion_layer"].visible = True
        sync_controls_to_active_dataset()

    @magicgui(
        annotation_key={"widget_type": "ComboBox", "choices": ["(none)"]},
        call_button="Remove Annotation From Zarr",
    )
    def remove_annotation_from_zarr_widget(annotation_key: str = "(none)"):
        if annotation_key == "(none)":
            return
        if not _confirm_data_delete(
            "Remove Annotation",
            f"Remove annotation '{annotation_key}' from the zarr?",
        ):
            return
        deleted = _run_with_busy_dialog(
            "Remove Annotation",
            "Deleting annotation from zarr...",
            lambda: delete_geojson_annotations(host_zarr_path, [annotation_key]),
        )
        if not deleted:
            QMessageBox.warning(None, "Remove Annotation", "No matching annotation was deleted.")
            return
        for dataset_state in datasets.values():
            dataset_state["dataset"].sdata = sd.read_zarr(dataset_state["dataset"].zarr_path)
        remove_annotation_shape_layers(deleted)
        sync_controls_to_active_dataset()

    def sync_controls_to_active_dataset():
        nonlocal ion_viewer_widget_syncing
        state = get_active_state()
        coreg_dataset = state["dataset"]
        if not state.get("ion_viewers"):
            state["ion_viewers"] = [{"name": "Viewer 1", "layer": state["ion_layer"], **copy_ion_view_state(state)}]
            state["active_ion_viewer_index"] = 0
        dataset_choice_to_key.clear()
        ordered_choices = []
        for item in datasets.values():
            choice = dataset_choice_text(item)
            dataset_choice_to_key[choice] = str(item["id"])
            ordered_choices.append(choice)
        remove_msi_dataset_from_zarr_widget.dataset.choices = ordered_choices if ordered_choices else [""]
        if remove_msi_dataset_from_zarr_widget.dataset.value not in remove_msi_dataset_from_zarr_widget.dataset.choices:
            remove_msi_dataset_from_zarr_widget.dataset.value = dataset_choice_text(state)
        copy_affine_to_target_widget.target_dataset.choices = ordered_choices
        rename_dataset_widget.display_name.value = str(state["label"])
        target_choices = [choice for choice in ordered_choices if dataset_choice_to_key.get(choice) != str(state["id"])]
        copy_affine_to_target_widget.target_dataset.choices = target_choices if target_choices else [""]
        if copy_affine_to_target_widget.target_dataset.value not in copy_affine_to_target_widget.target_dataset.choices:
            copy_affine_to_target_widget.target_dataset.value = copy_affine_to_target_widget.target_dataset.choices[0]
        viewer_choices = ion_viewer_choices(state)
        active_viewer_choice = ion_viewer_choice_text(state, int(state.get("active_ion_viewer_index", 0)))
        ion_viewer_widget_syncing = True
        try:
            ion_viewer_selector.viewer.choices = viewer_choices
            ion_viewer_selector.viewer.value = active_viewer_choice if active_viewer_choice in viewer_choices else viewer_choices[0]
        finally:
            ion_viewer_widget_syncing = False
        norm_mode = normalize_ion_display_mode(state.get("current_normalization_mode", "tic"))
        ion_display_options.normalization_mode.value = norm_mode
        ion_display_options.ratio_mz.value = str(state.get("current_ratio_mz", ""))
        ion_display_options.contrast_mode.value = str(state.get("current_contrast_mode", "percentile"))
        ion_display_options.contrast_percentiles.value = (
            float(state["current_contrast_low_pct"]),
            float(state["current_contrast_high_pct"]),
        )
        ion_display_options.absolute_low.value = f"{float(state.get('current_contrast_low', 0.0)):g}"
        ion_display_options.absolute_high.value = f"{float(state.get('current_contrast_high', 1.0)):g}"
        try:
            update_ion_status_label(state, np.asarray(state["ion_layer"].data, dtype=float))
        except Exception:
            pass
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
        schedule_scale_bar_update()

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
        _run_with_busy_dialog(
            "Add Optical Image",
            "Refreshing optical image layers...",
            refresh_datasets_after_reference_update,
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
        _run_with_busy_dialog(
            "Add H&E Image",
            "Refreshing H&E image layers...",
            refresh_datasets_after_reference_update,
        )
        if "hne" not in coreg_dataset.reference_image_keys:
            coreg_dataset.reference_image_keys.append("hne")
        add_or_update_reference_layer(coreg_dataset, "hne")
        refresh_if_threshold_choices(state)

    @magicgui(
        qptiff_level={"widget_type": "SpinBox", "min": 0, "max": 12, "step": 1},
        call_button="Add/Update H&E From QPTIFF",
    )
    def add_hne_from_qptiff(qptiff_level: int = 4):
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
        _run_with_busy_dialog(
            "Add H&E QPTIFF",
            "Refreshing H&E image layers...",
            refresh_datasets_after_reference_update,
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
        annotation_pyramid_level: int = 4,
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
        state["current_transform_is_initial_guess"] = False

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
                state["current_transform_is_initial_guess"] = False
        _run_with_busy_dialog("Save Registrations", "Saving all registrations...", save_all)

    @magicgui(call_button="Export Current View TIFF")
    def export_current_view_widget():
        default_name = f"{Path(get_active_state()['dataset'].zarr_path).stem}.tif"
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

    def section_panel(title: str, widgets: Iterable[QWidget], color: str) -> QWidget:
        panel = QWidget()
        panel.setObjectName(f"section_{sanitize_name(title)}")
        panel.setStyleSheet(
            f"QWidget#{panel.objectName()} {{ "
            f"background-color: {color}; "
            "border: 1px solid #4f5258; "
            "border-radius: 6px; "
            "}"
        )
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        label = QLabel(title)
        label.setStyleSheet("color: #ffffff; font-weight: 600;")
        layout.addWidget(label)
        for widget in widgets:
            layout.addWidget(widget)
        return panel

    controls_layout.addWidget(
        section_panel(
            "m/z selection / normalization",
            [spectrum_widget, mz_selector.native, ion_display_options.native, ion_status_label],
            "#2a3746",
        )
    )
    controls_layout.addWidget(
        section_panel(
            "Ion viewers",
            [ion_viewer_selector.native, add_ion_viewer_widget.native, remove_ion_viewer_widget.native],
            "#303d32",
        )
    )
    controls_layout.addWidget(
        section_panel(
            "Datasets and layers",
            [rename_dataset_widget.native, msi_layer_controls],
            "#42372a",
        )
    )
    controls_layout.addWidget(
        section_panel(
            "Masks and annotations",
            [
                roi_mask_controls.native,
                export_msi_from_roi_widget.native,
                annotation_display_controls.native,
                rename_selected_annotation_widget.native,
                view_selected_annotation_pixels_widget.native,
                export_selected_annotations_widget.native,
            ],
            "#3a3044",
        )
    )
    controls_layout.addWidget(
        section_panel(
            "View and export",
            [reset_canvas_view_widget.native, export_current_view_widget.native],
            "#2d2d30",
        )
    )
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
    alignment_dialog_container_layout.addWidget(copy_affine_to_target_widget.native)
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

    data_management_launcher = QWidget()
    data_management_launcher_layout = QVBoxLayout(data_management_launcher)
    data_management_launcher_layout.setContentsMargins(6, 6, 6, 6)
    data_management_launcher_layout.setSpacing(6)
    data_management_button = QPushButton("Open Data Management")
    data_management_launcher_layout.addWidget(data_management_button)
    data_management_launcher_layout.addWidget(QLabel("Remove MSI datasets or annotations from the zarr"))
    data_management_launcher_layout.addStretch(1)

    data_management_dialog = QDialog()
    data_management_dialog.setWindowTitle("Data Management")
    data_management_dialog.setModal(False)
    data_management_dialog.resize(500, 260)
    data_management_dialog_layout = QVBoxLayout(data_management_dialog)
    data_management_dialog_layout.setContentsMargins(8, 8, 8, 8)
    data_management_dialog_layout.setSpacing(8)
    data_management_dialog_layout.addWidget(remove_msi_dataset_from_zarr_widget.native)
    data_management_dialog_layout.addWidget(remove_annotation_from_zarr_widget.native)
    data_management_dialog_layout.addStretch(1)

    def open_data_management_dialog():
        sync_controls_to_active_dataset()
        data_management_dialog.show()
        data_management_dialog.raise_()
        data_management_dialog.activateWindow()

    data_management_button.clicked.connect(open_data_management_dialog)

    viewer.window.add_dock_widget(controls_scroll, area="right", name="Controls")
    viewer.window.add_dock_widget(add_data_launcher, area="left", name="Add Data")
    viewer.window.add_dock_widget(data_management_launcher, area="left", name="Data Management")
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
    schedule_scale_bar_update()
    napari.run()
    return viewer
