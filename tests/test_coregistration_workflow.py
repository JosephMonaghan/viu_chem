from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


sd = pytest.importorskip("spatialdata")
ad = pytest.importorskip("anndata")
gpd = pytest.importorskip("geopandas")
pytest.importorskip("shapely")

from shapely.geometry import box
from spatialdata.models import Image2DModel, ShapesModel, TableModel, get_channel_names
from spatialdata.transformations import Affine, Identity, get_transformation, set_transformation

from viu_chem.coreg_figures import get_coregistered_ion_image
from viu_chem.msi_coregistration import (
    CoregistrationDataset,
    _multiscale_image_levels,
    _normalize_registration_metric,
    _prepare_qptiff_image,
    _resolve_pyramid_level_index,
    _set_registration_metric,
    _xy_matrix_from_transform,
    add_reference_image,
    create_annotation_region_mask,
    convert_input_to_zarr,
    delete_msi_dataset,
    embed_msi_dataset,
    get_msi_table,
    import_geojson_annotations,
    list_coregistration_msi_datasets,
    rename_msi_dataset,
    rename_coordinate_system,
    rescale_registration_between_pyramid_levels,
    sample_reference_channel_values_at_msi_pixels,
    save_coregistration,
    save_reference_registration,
    set_reference_image_as_coordinate_system_anchor,
    sitk_affine_from_fixed_to_moving_matrix,
    sitk_transform_to_homogeneous_matrix,
)


def test_qptiff_singleton_channel_axis_is_restored_for_zarr_views():
    raw = np.ones((1, 5, 7), dtype=np.uint16)

    prepared, metadata = _prepare_qptiff_image(raw, "YX")

    assert prepared.shape == (5, 7, 1)
    assert metadata["source_axes"] == "CYX"
    assert metadata["source_channels"] == 1


def test_conversion_enables_thyra_auto_resampling_by_default(tmp_path: Path):
    calls = []

    def converter(**kwargs):
        calls.append(kwargs)
        return True

    output = convert_input_to_zarr(
        tmp_path / "input.imzML",
        tmp_path / "output.zarr",
        converter=converter,
    )

    assert output == tmp_path / "output.zarr"
    assert calls[0]["resampling_config"] == {}


def test_conversion_can_preserve_the_raw_union_mass_axis(tmp_path: Path):
    calls = []

    def converter(**kwargs):
        calls.append(kwargs)
        return True

    convert_input_to_zarr(
        tmp_path / "input.imzML",
        tmp_path / "output.zarr",
        converter=converter,
        resample=False,
    )

    assert "resampling_config" not in calls[0]
    with pytest.raises(ValueError, match="requires resample=True"):
        convert_input_to_zarr(
            tmp_path / "input.imzML",
            converter=converter,
            resample=False,
            resampling_config={"method": "nearest_neighbor"},
        )


def test_pyramid_level_minus_one_resolves_to_highest_available_level():
    assert _resolve_pyramid_level_index(-1, 5) == 4
    assert _resolve_pyramid_level_index(2, 5) == 2
    assert _resolve_pyramid_level_index(-1, 1) == 0

    with pytest.raises(ValueError, match="valid levels are 0 through 4"):
        _resolve_pyramid_level_index(5, 5)


def test_registration_metric_names_support_mi_and_normalized_cross_correlation():
    assert _normalize_registration_metric("MI") == "Mutual information"
    assert _normalize_registration_metric("mutual_information") == "Mutual information"
    assert _normalize_registration_metric("NCC") == "Normalized cross-correlation"
    assert _normalize_registration_metric("normalized cross correlation") == "Normalized cross-correlation"

    with pytest.raises(ValueError, match="Unsupported registration metric"):
        _normalize_registration_metric("mean squared error")


def test_registration_metric_configures_mutual_information_or_correlation():
    class FakeRegistration:
        def __init__(self):
            self.metric = None

        def SetMetricAsMattesMutualInformation(self, *, numberOfHistogramBins):
            self.metric = ("mi", numberOfHistogramBins)

        def SetMetricAsCorrelation(self):
            self.metric = ("ncc", None)

    registration = FakeRegistration()
    assert _set_registration_metric(registration, "MI", histogram_bins=64) == "Mutual information"
    assert registration.metric == ("mi", 64)

    assert _set_registration_metric(registration, "NCC", histogram_bins=64) == "Normalized cross-correlation"
    assert registration.metric == ("ncc", None)


def _pixel_shapes(name: str = "pixels"):
    shapes = gpd.GeoDataFrame(
        {"instance_id": [0, 1, 2, 3]},
        geometry=[
            box(-0.5, -0.5, 0.5, 0.5),
            box(0.5, -0.5, 1.5, 0.5),
            box(-0.5, 0.5, 0.5, 1.5),
            box(0.5, 0.5, 1.5, 1.5),
        ],
    ).set_index("instance_id")
    shapes.index.name = "instance_id"
    return ShapesModel.parse(shapes)


def _msi_table(*, region: str = "pixels", tic_key: str = "msi_tic"):
    obs = pd.DataFrame(
        {
            "x": [0, 1, 0, 1],
            "y": [0, 0, 1, 1],
            "region": pd.Categorical([region] * 4),
            "instance_id": [0, 1, 2, 3],
        },
        index=[f"pixel_{idx}" for idx in range(4)],
    )
    var = pd.DataFrame({"mz": [100.0, 200.0]}, index=["mz_100", "mz_200"])
    table = TableModel.parse(
        ad.AnnData(
            X=np.array(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0],
                    [7.0, 8.0],
                ]
            ),
            obs=obs,
            var=var,
        ),
        region=region,
        region_key="region",
        instance_key="instance_id",
    )
    table.uns["coregistration_dataset_label"] = "msi"
    table.uns["coregistration_display_name"] = "MSI"
    table.uns["coregistration_tic_key"] = tic_key
    table.uns["coregistration_pixel_shape_keys"] = [region]
    return table


def _write_coregistration_store(
    path: Path,
    *,
    table_key: str = "msi",
    tic_key: str = "msi_tic",
    pixel_key: str = "pixels",
    include_reference: bool = True,
    include_roi: bool = True,
) -> Path:
    tic = Image2DModel.parse(
        np.array([[[10.0, 20.0], [30.0, 40.0]]]),
        dims=("c", "y", "x"),
        c_coords=["TIC"],
    )
    images = {tic_key: tic}
    if include_reference:
        reference = Image2DModel.parse(
            np.array(
                [
                    [
                        [10.0, 20.0, 30.0],
                        [40.0, 50.0, 60.0],
                        [70.0, 80.0, 90.0],
                    ]
                ]
            ),
            dims=("c", "y", "x"),
            c_coords=["reference"],
        )
        set_transformation(reference, Identity(), to_coordinate_system="registered")
        images["hne"] = reference

    shapes = {pixel_key: _pixel_shapes(pixel_key)}
    if include_roi:
        roi = ShapesModel.parse(
            gpd.GeoDataFrame(
                {"_annotation_label": ["thin_roi"]},
                geometry=[box(0.4, -0.25, 0.6, 0.25)],
            )
        )
        set_transformation(roi, Identity(), to_coordinate_system="registered")
        shapes["anno_thin_roi"] = roi

    table = _msi_table(region=pixel_key, tic_key=tic_key)
    table.uns["coregistration_dataset_label"] = table_key
    table.uns["coregistration_display_name"] = table_key.upper()
    sdata = sd.SpatialData(images=images, shapes=shapes, tables={table_key: table})
    sdata.write(path)
    return path


@pytest.fixture
def coregistration_store(tmp_path: Path) -> Path:
    return _write_coregistration_store(tmp_path / "coregistration.zarr")


def test_dataset_reconstructs_raw_and_tic_normalized_ion_images(coregistration_store: Path):
    dataset = CoregistrationDataset(coregistration_store)

    np.testing.assert_array_equal(
        dataset.reconstruct_ion_image(0, normalize_to_tic=False),
        np.array([[1.0, 3.0], [5.0, 7.0]]),
    )
    np.testing.assert_allclose(
        dataset.reconstruct_ion_image(0, normalize_to_tic=True),
        np.array([[1.0 / 10.0, 3.0 / 20.0], [5.0 / 30.0, 7.0 / 40.0]]),
    )
    np.testing.assert_array_equal(dataset.find_feature_indices_from_mz(100.0004, 5.0), np.array([0]))


def test_dataset_can_reuse_loaded_spatialdata_without_reading_store_again(coregistration_store: Path, monkeypatch):
    shared_sdata = sd.read_zarr(coregistration_store)

    def unexpected_read(*_args, **_kwargs):
        raise AssertionError("shared SpatialData should avoid another read_zarr call")

    monkeypatch.setattr("viu_chem.msi_coregistration.sd.read_zarr", unexpected_read)
    dataset = CoregistrationDataset(coregistration_store, sdata=shared_sdata)

    assert dataset.sdata is shared_sdata
    assert dataset.table_key == "msi"


def test_spectrum_plot_indices_are_bounded_and_keep_strongest_local_peaks():
    intensity = np.zeros(30, dtype=float)
    intensity[[2, 5, 8, 11, 14, 17, 20, 23]] = [1, 8, 2, 7, 3, 6, 4, 5]

    indices = CoregistrationDataset.spectrum_plot_indices(intensity, max_peaks=4)

    np.testing.assert_array_equal(indices, np.array([5, 11, 17, 23]))


def test_sorted_mass_axis_lookups_use_ppm_window_boundaries(coregistration_store: Path):
    dataset = CoregistrationDataset(coregistration_store)
    dataset.mz_values = np.array([99.9990, 100.0, 100.0004, 100.0010])
    dataset._mz_values_are_sorted = True

    np.testing.assert_array_equal(
        dataset.find_feature_indices_from_mz(100.0, 5.0),
        np.array([1, 2]),
    )
    idx, ppm_error = dataset.find_feature_idx_from_mz(100.0003, 5.0)
    assert idx == 2
    assert ppm_error == pytest.approx(1.0, rel=1e-3)


def test_unsorted_legacy_mass_axis_lookup_keeps_compatibility(coregistration_store: Path):
    dataset = CoregistrationDataset(coregistration_store)
    dataset.mz_values = np.array([200.0, 100.0004, 100.0])
    dataset._mz_values_are_sorted = False

    np.testing.assert_array_equal(
        dataset.find_feature_indices_from_mz(100.0, 5.0),
        np.array([1, 2]),
    )
    idx, _ = dataset.find_feature_idx_from_mz(100.0001, 5.0)
    assert idx == 2


def test_region_spectra_can_normalize_each_pixel_to_a_reference_mz(coregistration_store: Path):
    dataset = CoregistrationDataset(coregistration_store)
    selected = np.ones(4, dtype=bool)

    summary = dataset.summarize_region_spectra(
        selected,
        normalize_to=200.0,
        normalize_to_ppm_tolerance=5.0,
    )

    normalized = np.array(
        [
            [1.0 / 2.0, 1.0],
            [3.0 / 4.0, 1.0],
            [5.0 / 6.0, 1.0],
            [7.0 / 8.0, 1.0],
        ]
    )
    np.testing.assert_allclose(summary["mean_intensity"], normalized.mean(axis=0))
    np.testing.assert_allclose(summary["std_intensity"], normalized.std(axis=0, ddof=0))
    assert summary["n_spectra"] == 4

    with pytest.raises(ValueError, match="No MSI features found"):
        dataset.summarize_region_spectra(
            selected,
            normalize_to=300.0,
            normalize_to_ppm_tolerance=5.0,
        )


def test_interactive_spectrum_masks_select_pixels_and_regions(coregistration_store: Path):
    dataset = CoregistrationDataset(coregistration_store)

    np.testing.assert_array_equal(
        dataset.spectrum_mask_at_image_position((0.1, 0.2)),
        np.array([True, False, False, False]),
    )
    assert not dataset.spectrum_mask_at_image_position((2.0, 2.0)).any()

    region = np.array(
        [
            [-0.25, -0.25],
            [-0.25, 1.25],
            [0.25, 1.25],
            [0.25, -0.25],
        ]
    )
    np.testing.assert_array_equal(
        dataset.spectrum_mask_in_image_regions([region]),
        np.array([True, True, False, False]),
    )
    mean_spectrum, count = dataset.mean_spectrum_for_selection(np.array([True, True, False, False]))
    np.testing.assert_allclose(mean_spectrum, np.array([2.0, 3.0]))
    assert count == 2


def test_save_registration_persists_one_affine_for_tic_and_pixel_shapes(coregistration_store: Path):
    transform_xy = np.array(
        [
            [2.0, 0.1, 5.0],
            [0.2, 3.0, 7.0],
            [0.0, 0.0, 1.0],
        ]
    )

    save_coregistration(coregistration_store, transform_xy)
    reloaded = sd.read_zarr(coregistration_store)

    tic_transform = get_transformation(reloaded.images["msi_tic"], to_coordinate_system="registered")
    pixel_transform = get_transformation(reloaded.shapes["pixels"], to_coordinate_system="registered")
    reference_transform = get_transformation(reloaded.images["hne"], to_coordinate_system="registered")
    np.testing.assert_allclose(_xy_matrix_from_transform(tic_transform), transform_xy)
    np.testing.assert_allclose(_xy_matrix_from_transform(pixel_transform), transform_xy)
    np.testing.assert_allclose(_xy_matrix_from_transform(reference_transform), np.eye(3))

    dataset = CoregistrationDataset(coregistration_store)
    loaded_transform, found = dataset.load_saved_registration_if_available()
    assert found
    np.testing.assert_allclose(loaded_transform, transform_xy)


def test_simpleitk_affine_conversion_round_trips_full_affine():
    pytest.importorskip("SimpleITK")
    transform_xy = np.array(
        [
            [1.2, 0.15, 12.0],
            [-0.05, 0.9, -8.0],
            [0.0, 0.0, 1.0],
        ]
    )

    sitk_transform = sitk_affine_from_fixed_to_moving_matrix(transform_xy)

    np.testing.assert_allclose(sitk_transform_to_homogeneous_matrix(sitk_transform), transform_xy)


def test_reference_sampling_uses_registered_msi_pixel_footprints(coregistration_store: Path):
    save_coregistration(coregistration_store, np.eye(3))

    values = sample_reference_channel_values_at_msi_pixels(
        coregistration_store,
        reference_key="hne",
        channel_index=0,
    )

    np.testing.assert_allclose(values, np.array([10.0, 20.0, 40.0, 50.0]))


def test_annotation_mask_preserves_center_and_intersection_semantics(coregistration_store: Path):
    save_coregistration(coregistration_store, np.eye(3))

    center_mask = create_annotation_region_mask(
        coregistration_store,
        "anno_thin_roi",
        inclusion_mode="center",
    )
    intersection_mask = create_annotation_region_mask(
        coregistration_store,
        "anno_thin_roi",
        inclusion_mode="intersects",
    )

    np.testing.assert_array_equal(center_mask, np.array([False, False, False, False]))
    np.testing.assert_array_equal(intersection_mask, np.array([True, True, False, False]))


def test_public_ion_image_resampling_preserves_registered_translation(coregistration_store: Path):
    save_coregistration(
        coregistration_store,
        np.array(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )

    layer = get_coregistered_ion_image(
        coregistration_store,
        100.0,
        normalize_to_tic=False,
        mask_low=False,
        resample_order=0,
    )

    np.testing.assert_allclose(
        np.asarray(layer.data),
        np.array(
            [
                [np.nan, 1.0, 3.0],
                [np.nan, 5.0, 7.0],
                [np.nan, np.nan, np.nan],
            ]
        ),
        equal_nan=True,
    )


def test_reference_and_geojson_ingestion_preserves_geometry_and_anchor(tmp_path: Path):
    store = _write_coregistration_store(
        tmp_path / "ingestion.zarr",
        include_reference=False,
        include_roi=False,
    )
    reference_path = tmp_path / "reference.tif"
    import tifffile

    grayscale = np.arange(20, dtype=np.uint16).reshape(4, 5)
    tifffile.imwrite(reference_path, np.repeat(grayscale[..., None], 3, axis=-1), photometric="rgb")
    add_reference_image(store, reference_path, key="hne")

    geojson_path = tmp_path / "regions.geojson"
    geojson_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"name": "tumor"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                        },
                    }
                ],
            }
        )
    )

    imported = import_geojson_annotations(
        store,
        [geojson_path],
        annotation_scale_x=2.0,
        annotation_scale_y=2.0,
        annotation_translate_x=3.0,
        annotation_translate_y=4.0,
    )

    assert imported == ["anno_regions"]
    reloaded = sd.read_zarr(store)
    assert tuple(reloaded.images["hne"].shape[-2:]) == (4, 5)
    np.testing.assert_allclose(reloaded.shapes["anno_regions"].total_bounds, np.array([3.0, 4.0, 5.0, 6.0]))
    annotation_transform = get_transformation(
        reloaded.shapes["anno_regions"],
        to_coordinate_system="registered",
    )
    np.testing.assert_allclose(_xy_matrix_from_transform(annotation_transform), np.eye(3))


def test_grayscale_reference_ingestion_is_supported(tmp_path: Path):
    store = _write_coregistration_store(
        tmp_path / "grayscale.zarr",
        include_reference=False,
        include_roi=False,
    )
    reference_path = tmp_path / "grayscale.tif"
    import tifffile

    tifffile.imwrite(reference_path, np.arange(20, dtype=np.uint16).reshape(4, 5))

    add_reference_image(store, reference_path, key="hne")
    image = sd.read_zarr(store).images["hne"]
    assert image.dims == ("c", "y", "x")
    assert image.shape == (1, 4, 5)
    assert list(get_channel_names(image)) == ["image"]


def test_qptiff_reference_ingestion_preserves_native_lazy_pyramid(tmp_path: Path):
    import tifffile

    store = _write_coregistration_store(
        tmp_path / "pyramidal.zarr",
        include_reference=False,
        include_roi=False,
    )
    reference_path = tmp_path / "fluorescence.qptiff"
    full = np.arange(3 * 64 * 80, dtype=np.uint16).reshape(3, 64, 80)
    with tifffile.TiffWriter(reference_path, ome=False) as tif:
        tif.write(full, metadata={"axes": "CYX"}, tile=(16, 16), subifds=2)
        tif.write(full[:, ::2, ::2], metadata={"axes": "CYX"}, tile=(16, 16), subfiletype=1)
        tif.write(full[:, ::4, ::4], metadata={"axes": "CYX"}, tile=(16, 16), subfiletype=1)

    dataset = add_reference_image(store, reference_path, key="hne", qptiff_level=None)
    image = dataset.sdata.images["hne"]
    levels = _multiscale_image_levels(image)

    assert [level.shape for level in levels] == [(3, 64, 80), (3, 32, 40), (3, 16, 20)]
    assert all(hasattr(level.data, "chunks") for level in levels)
    assert image.attrs["image_source"] == "qptiff_pyramid_multiscale"
    assert image.attrs["pyramid_level_shapes_yx"] == [[64, 80], [32, 40], [16, 20]]
    np.testing.assert_array_equal(np.asarray(levels[0][0, :2, :3]), full[0, :2, :3])


def test_scn_ingestion_selects_largest_scene_and_accepts_custom_layer_name(tmp_path: Path):
    import tifffile

    store = _write_coregistration_store(
        tmp_path / "scn.zarr",
        include_reference=False,
        include_roi=False,
    )
    slide_path = tmp_path / "histology.scn"
    thumbnail = np.zeros((8, 10, 3), dtype=np.uint8)
    full = np.arange(32 * 40 * 3, dtype=np.uint16).reshape(32, 40, 3)
    with tifffile.TiffWriter(slide_path) as tif:
        tif.write(thumbnail, photometric="rgb")
        tif.write(full, photometric="rgb", tile=(16, 16), subifds=1)
        tif.write(full[::2, ::2], photometric="rgb", tile=(16, 16), subfiletype=1)

    dataset = add_reference_image(
        store,
        slide_path,
        key="Histology scan 1",
        image_type="H&E / brightfield",
    )
    image = dataset.sdata.images["histology_scan_1"]
    levels = _multiscale_image_levels(image)

    assert [level.shape for level in levels] == [(3, 32, 40), (3, 16, 20)]
    assert image.attrs["tiff_series"] == 1
    assert image.attrs["image_type"] == "H&E / brightfield"
    assert "registered" in dataset.sdata.coordinate_systems


def test_reference_image_can_rebase_common_coordinate_system(tmp_path: Path):
    store = _write_coregistration_store(tmp_path / "rebase.zarr")
    sdata = sd.read_zarr(store)
    anchor_xy = np.array([[2.0, 0.0, 10.0], [0.0, 2.0, -4.0], [0.0, 0.0, 1.0]])
    msi_xy = np.array([[3.0, 0.0, 13.0], [0.0, 3.0, 2.0], [0.0, 0.0, 1.0]])
    set_transformation(
        sdata.images["hne"],
        Affine(anchor_xy, input_axes=("x", "y"), output_axes=("x", "y")),
        to_coordinate_system="registered",
    )
    set_transformation(
        sdata.images["msi_tic"],
        Affine(msi_xy, input_axes=("x", "y"), output_axes=("x", "y")),
        to_coordinate_system="registered",
    )
    sdata.write_transformations("hne")
    sdata.write_transformations("msi_tic")

    set_reference_image_as_coordinate_system_anchor(store, "hne")

    reloaded = sd.read_zarr(store)
    np.testing.assert_allclose(
        _xy_matrix_from_transform(get_transformation(reloaded.images["hne"], to_coordinate_system="registered")),
        np.eye(3),
    )
    np.testing.assert_allclose(
        _xy_matrix_from_transform(get_transformation(reloaded.images["msi_tic"], to_coordinate_system="registered")),
        np.linalg.inv(anchor_xy) @ msi_xy,
    )


def test_coordinate_system_can_be_renamed_without_changing_transforms(tmp_path: Path):
    store = _write_coregistration_store(tmp_path / "rename-coordinate-system.zarr")
    before = sd.read_zarr(store)
    expected = {
        name: _xy_matrix_from_transform(get_transformation(element, to_coordinate_system="registered"))
        for _kind, name, element in before.gen_spatial_elements()
        if "registered" in get_transformation(element, get_all=True)
    }

    changed = rename_coordinate_system(store, "registered", "IF aligned")

    reloaded = sd.read_zarr(store)
    assert changed == len(expected)
    assert "if_aligned" in reloaded.coordinate_systems
    assert "registered" not in reloaded.coordinate_systems
    for _kind, name, element in reloaded.gen_spatial_elements():
        if name in expected:
            np.testing.assert_allclose(
                _xy_matrix_from_transform(get_transformation(element, to_coordinate_system="if_aligned")),
                expected[name],
            )


def test_reference_registration_affine_is_persisted(tmp_path: Path):
    store = _write_coregistration_store(tmp_path / "reference-registration.zarr")
    transform_xy = np.array(
        [[0.9, -0.2, 14.0], [0.1, 1.1, -8.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )

    save_reference_registration(store, "hne", transform_xy)

    reloaded = sd.read_zarr(store)
    np.testing.assert_allclose(
        _xy_matrix_from_transform(
            get_transformation(reloaded.images["hne"], to_coordinate_system="registered")
        ),
        transform_xy,
    )


def test_registration_affine_can_be_converted_between_reference_pyramid_levels(tmp_path: Path):
    import tifffile

    store = _write_coregistration_store(
        tmp_path / "affine-pyramid.zarr",
        include_reference=False,
        include_roi=False,
    )
    reference_path = tmp_path / "affine-fluorescence.qptiff"
    full = np.ones((1, 65, 81), dtype=np.uint16)
    coarse = np.full((1, 17, 21), 7, dtype=np.uint16)
    with tifffile.TiffWriter(reference_path, ome=False) as tif:
        tif.write(full, metadata={"axes": "CYX"}, tile=(16, 16), subifds=1)
        tif.write(coarse, metadata={"axes": "CYX"}, tile=(16, 16), subfiletype=1)

    image = add_reference_image(store, reference_path, key="hne", qptiff_level=None).sdata.images["hne"]
    level_transform = np.array(
        [[1.2, -0.1, 3.0], [0.2, 0.9, -4.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    converted, scales = rescale_registration_between_pyramid_levels(
        level_transform,
        image,
        source_level=1,
        target_level=0,
    )

    expected_scale_x = 81 / 21
    expected_scale_y = 65 / 17
    np.testing.assert_allclose(scales, (expected_scale_x, expected_scale_y))
    np.testing.assert_allclose(
        converted,
        np.diag([expected_scale_x, expected_scale_y, 1.0]) @ level_transform,
    )
    round_trip, _ = rescale_registration_between_pyramid_levels(
        converted,
        image,
        source_level=0,
        target_level=1,
    )
    np.testing.assert_allclose(round_trip, level_transform)

    full_values = sample_reference_channel_values_at_msi_pixels(
        CoregistrationDataset(store),
        reference_key="hne",
        channel_index=0,
        transform_xy=np.eye(3),
        reference_pyramid_level=0,
    )
    coarse_values = sample_reference_channel_values_at_msi_pixels(
        CoregistrationDataset(store),
        reference_key="hne",
        channel_index=0,
        transform_xy=np.eye(3),
        reference_pyramid_level=1,
    )
    np.testing.assert_allclose(full_values, 1.0)
    np.testing.assert_allclose(coarse_values, 7.0)

    with pytest.raises(ValueError, match="valid levels are 0 through 1"):
        rescale_registration_between_pyramid_levels(
            level_transform,
            image,
            source_level=4,
            target_level=0,
        )


def test_embedding_another_msi_dataset_preserves_current_selection_contract(tmp_path: Path):
    host = _write_coregistration_store(tmp_path / "host.zarr")
    source = _write_coregistration_store(
        tmp_path / "source.zarr",
        table_key="source",
        tic_key="source_tic",
        pixel_key="source_pixels",
        include_reference=False,
        include_roi=False,
    )

    result = embed_msi_dataset(host, source, dataset_label="Negative Mode")

    assert result["table_key"] == "negative_mode"
    assert result["tic_key"] == "negative_mode_tic"
    specs = list_coregistration_msi_datasets(host)
    assert {(spec["table_key"], spec["tic_key"]) for spec in specs} == {
        ("msi", "msi_tic"),
        ("negative_mode", "negative_mode_tic"),
    }
    embedded = CoregistrationDataset(host, table_key="negative_mode")
    np.testing.assert_array_equal(
        embedded.reconstruct_ion_image(1, normalize_to_tic=False),
        np.array([[2.0, 4.0], [6.0, 8.0]]),
    )

    assert rename_msi_dataset(host, table_key="negative_mode", display_name="Negative renamed") == "Negative renamed"
    renamed = next(spec for spec in list_coregistration_msi_datasets(host) if spec["table_key"] == "negative_mode")
    assert renamed["display_name"] == "Negative renamed"

    deleted = delete_msi_dataset(host, table_key="negative_mode")
    assert deleted == {
        "tables": ["negative_mode"],
        "images": ["negative_mode_tic"],
        "shapes": result["pixel_shape_keys"],
    }
    assert {spec["table_key"] for spec in list_coregistration_msi_datasets(host)} == {"msi"}


def test_embedding_aligned_imzml_can_skip_thyra_resampling(tmp_path: Path):
    host = _write_coregistration_store(tmp_path / "host.zarr")
    converter_calls = []

    def converter(**kwargs):
        converter_calls.append(kwargs)
        _write_coregistration_store(
            Path(kwargs["output_path"]),
            table_key="source",
            tic_key="source_tic",
            pixel_key="source_pixels",
            include_reference=False,
            include_roi=False,
        )
        return True

    result = embed_msi_dataset(
        host,
        tmp_path / "aligned.imzML",
        dataset_label="Aligned MSI",
        converter=converter,
        skip_resampling=True,
    )

    assert "resampling_config" not in converter_calls[0]
    assert result["table_key"] == "aligned_msi"
    assert result["tic_key"] == "aligned_msi_tic"


def test_get_msi_table_resolves_display_name_with_one_store_read(tmp_path: Path, monkeypatch):
    host = _write_coregistration_store(tmp_path / "host.zarr")
    source = _write_coregistration_store(
        tmp_path / "source.zarr",
        table_key="source",
        tic_key="source_tic",
        pixel_key="source_pixels",
        include_reference=False,
        include_roi=False,
    )
    embed_msi_dataset(host, source, dataset_label="nano-DESI (Positive)")
    real_read_zarr = sd.read_zarr
    read_count = 0

    def counted_read_zarr(*args, **kwargs):
        nonlocal read_count
        read_count += 1
        return real_read_zarr(*args, **kwargs)

    monkeypatch.setattr("viu_chem.msi_coregistration.sd.read_zarr", counted_read_zarr)
    table = get_msi_table(host, "NANO-desi (positive)")

    assert table.uns["coregistration_display_name"] == "nano-DESI (Positive)"
    np.testing.assert_array_equal(table.var["mz"].values, np.array([100.0, 200.0]))
    assert read_count == 1


def test_embedding_preserves_standard_spatialdata_table_region_relationship(tmp_path: Path):
    host = _write_coregistration_store(tmp_path / "host.zarr")
    source = _write_coregistration_store(
        tmp_path / "source.zarr",
        table_key="source",
        tic_key="source_tic",
        pixel_key="source_pixels",
        include_reference=False,
        include_roi=False,
    )

    result = embed_msi_dataset(host, source, dataset_label="Negative Mode")
    reloaded = sd.read_zarr(host)
    table = reloaded.tables[result["table_key"]]

    assert set(sd.SpatialData.get_annotated_regions(table)) == set(result["pixel_shape_keys"])
    assert set(table.obs["region"].astype(str)) == set(result["pixel_shape_keys"])
