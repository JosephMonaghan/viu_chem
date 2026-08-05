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
from spatialdata.transformations import Identity, get_transformation, set_transformation

from viu_chem.coreg_figures import get_coregistered_ion_image
from viu_chem.msi_coregistration import (
    CoregistrationDataset,
    _xy_matrix_from_transform,
    add_reference_image,
    create_annotation_region_mask,
    delete_msi_dataset,
    embed_msi_dataset,
    import_geojson_annotations,
    list_coregistration_msi_datasets,
    rename_msi_dataset,
    sample_reference_channel_values_at_msi_pixels,
    save_coregistration,
    sitk_affine_from_fixed_to_moving_matrix,
    sitk_transform_to_homogeneous_matrix,
)


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
