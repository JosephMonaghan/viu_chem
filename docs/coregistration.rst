Coregistration Workflow
=======================

The coregistration tools combine MSI data, reference images such as H&E or IF,
ROI annotations, and saved affine transforms in one SpatialData ``.zarr``. The
main entry points live in ``viu_chem.msi_coregistration``, the napari interface
is in ``viu_chem.coreg_gui``, and publication-style overlays are in
``viu_chem.coreg_figures``.

Install the optional dependencies before using these workflows:

.. code-block:: bash

   pip install "viu-chem[coregistration]"

Prepare a Coregistration Zarr
-----------------------------

Use ``prepare_coregistration_zarr`` when you want to create or update a
SpatialData zarr from an MSI input and optional reference images.

.. code-block:: python

   from pathlib import Path
   from viu_chem.msi_coregistration import prepare_coregistration_zarr

   zarr_path = prepare_coregistration_zarr(
       input_path=Path("sample.imzML"),
       zarr_path=Path("sample.zarr"),
       hne_image_path=Path("sample_hne.tif"),
       annotation_paths=[Path("sample_annotations.geojson")],
   )

If the zarr already exists, you can add data incrementally:

.. code-block:: python

   from viu_chem.msi_coregistration import (
       add_reference_image,
       embed_msi_dataset,
       import_geojson_annotations,
   )

   add_reference_image("sample.zarr", "sample_hne.tif", key="hne")
   add_reference_image("sample.zarr", "sample_if.qptiff", key="optical", qptiff_level=2)

   embed_msi_dataset(
       "sample.zarr",
       "negative_mode.imzML",
       dataset_label="nanoDESI Negative",
   )

   import_geojson_annotations(
       "sample.zarr",
       ["annotations.geojson"],
       target_image="hne",
       object_mode="annotations_only",
       annotation_pyramid_level=4,
   )

The dataset selector used throughout the API accepts the dataset display name,
internal label, table key, or TIC image key:

.. code-block:: python

   from viu_chem.msi_coregistration import list_coregistration_msi_datasets

   for spec in list_coregistration_msi_datasets("sample.zarr"):
       print(spec["display_name"], spec["table_key"], spec["tic_key"])

Launch the GUI
--------------

The napari GUI is the easiest way to inspect and save the registration. It can
open an existing zarr or convert an input path first.

.. code-block:: python

   from viu_chem import coreg_gui

   coreg_gui.launch_coregistration_gui(zarr_path="sample.zarr")

Inside the GUI, use the alignment tools to adjust landmarks, optimize the
affine registration, and save the active transform. The saved transform is
written to the SpatialData coordinate system named ``registered`` by default.
When multiple MSI datasets are embedded, each dataset can have its own saved
MSI-to-reference transform.

Reference and MSI Access
------------------------

``CoregistrationDataset`` is a small convenience wrapper around the zarr. It
loads the selected MSI table, TIC image, coordinates, m/z values, and saved
registration transform.

.. code-block:: python

   from viu_chem.msi_coregistration import (
       CoregistrationDataset,
       list_coregistration_msi_datasets,
   )

   specs = list_coregistration_msi_datasets("sample.zarr")
   negative = next(spec for spec in specs if spec["display_name"] == "nanoDESI Negative")

   dataset = CoregistrationDataset(
       "sample.zarr",
       table_key=negative["table_key"],
       tic_key=negative["tic_key"],
   )

   ion_image = dataset.reconstruct_ion_image(
       dataset.find_feature_indices_from_mz(611.1447, ppm_tolerance=5),
       normalize_to_tic=True,
   )

   transform_xy, found = dataset.load_saved_registration_if_available()
   print(found, transform_xy)

The public helpers can also sample reference channels at MSI pixels:

.. code-block:: python

   from viu_chem.msi_coregistration import sample_reference_channel_values_at_msi_pixels

   values = sample_reference_channel_values_at_msi_pixels(
       "sample.zarr",
       reference_key="hne",
       channel_index=0,
       msi_dataset="nanoDESI Negative",
   )

Annotations and ROI Masks
-------------------------

Imported GeoJSON annotations are stored as SpatialData shapes. To convert one
annotation or one annotation label into a boolean MSI pixel mask, use
``create_annotation_region_mask``.

.. code-block:: python

   from viu_chem.msi_coregistration import (
       create_annotation_region_mask,
       summarize_annotation_region_spectra,
   )

   roi_mask = create_annotation_region_mask(
       "sample.zarr",
       "anno_tumor_regions",
       region_label="Tumor",
       msi_dataset="nanoDESI Negative",
       inclusion_mode="center",
   )

   summary = summarize_annotation_region_spectra(
       "sample.zarr",
       "anno_tumor_regions",
       region_label="Tumor",
       msi_dataset="nanoDESI Negative",
       normalize_to_tic=True,
   )

``summarize_msi_pixel_mask_spectra`` can summarize any boolean mask that is the
same length as the number of MSI spectra.

Threshold Masks and Prefilters
------------------------------

MSI thresholds split pixels into below and above groups for one m/z feature.
Supply exactly one of ``threshold`` or ``percentile``.

.. code-block:: python

   from viu_chem.msi_coregistration import create_msi_threshold_mask

   msi_mask = create_msi_threshold_mask(
       "sample.zarr",
       target_mz=611.1447,
       ppm_tolerance=5,
       percentile=90,
       msi_dataset="nanoDESI Negative",
       prefilter_mask=roi_mask,
   )

   print(msi_mask.threshold)
   print(msi_mask.above_mask.sum(), msi_mask.below_mask.sum())

Reference thresholds split pixels by sampled IF/H&E intensity at MSI pixel
locations.

.. code-block:: python

   from viu_chem.msi_coregistration import create_reference_threshold_mask

   if_mask = create_reference_threshold_mask(
       "sample.zarr",
       reference_key="hne",
       channel_index=0,
       percentile=75,
       msi_dataset="nanoDESI Negative",
       prefilter_mask=roi_mask,
   )

For percentile thresholds, the percentile is computed only from finite pixels
inside ``prefilter_mask``. Pixels outside the prefilter are excluded from both
``below_mask`` and ``above_mask``. For absolute thresholds, the threshold value
is fixed, but the prefilter still controls which pixels are eligible for either
output group.

Use the summary wrappers when you want mean spectra for the thresholded groups:

.. code-block:: python

   from viu_chem.msi_coregistration import summarize_reference_threshold_spectra

   spectra = summarize_reference_threshold_spectra(
       "sample.zarr",
       reference_key="hne",
       channel_index=0,
       percentile=75,
       msi_dataset="nanoDESI Negative",
       prefilter_mask=roi_mask,
   )

   above_mean = spectra.above_summary["mean_intensity"]
   mz = spectra.above_summary["mz"]

Threshold Masks as Saved Annotations
------------------------------------

The GUI threshold tools can preview MSI or IF thresholds and save the resulting
below/above regions as annotations. The same behavior is available from Python.

.. code-block:: python

   from viu_chem.msi_coregistration import create_reference_threshold_annotation

   key = create_reference_threshold_annotation(
       "sample.zarr",
       reference_key="hne",
       channel_index=0,
       threshold=if_mask.threshold,
       table_key=negative["table_key"],
       tic_key=negative["tic_key"],
       prefilter_mask=roi_mask,
       prefilter_shape_key="anno_tumor_regions",
       prefilter_region_label="Tumor",
       annotation_name="hne_ch1_high_low",
       annotation_label="HNE ch1",
   )

For MSI intensity thresholds, use ``create_msi_threshold_annotation``. For
mapping an MSI threshold from one embedded dataset onto another, use
``create_pooled_msi_threshold_annotation``.

Colocalization and Correlation
------------------------------

Coregistration also includes helpers to search for MSI features that colocalize
with an MSI reference feature or with IF/H&E reference channels.

.. code-block:: python

   from viu_chem.msi_coregistration import (
       colocalized_msi_features,
       correlate_msi_features_with_reference_channels,
   )

   msi_hits = colocalized_msi_features(
       "sample.zarr",
       reference_mz=611.1447,
       msi_dataset="nanoDESI Negative",
       n=25,
       sort_by="correlation",
       threshold="median",
       prefilter_mask=roi_mask,
   )

   if_hits = correlate_msi_features_with_reference_channels(
       "sample.zarr",
       reference_key="hne",
       channel_indices=[0, 1],
       msi_dataset="nanoDESI Negative",
       n=25,
       sort_by="correlation",
       prefilter_mask=roi_mask,
   )

``colocalized_msi_features_between_datasets`` performs the same search when
the reference m/z is in one embedded MSI dataset and the candidate features are
in another.

Visualize Coregistered Data
---------------------------

Use ``coreg_figures`` to make reusable arrays or export complete figures. The
``get_coregistered_*`` functions return ``CoregisteredImage`` objects with
``data`` and display limits.

.. code-block:: python

   import matplotlib.pyplot as plt
   from viu_chem.coreg_figures import (
       get_coregistered_reference_image,
       get_coregistered_msi_mask_image,
   )

   ref = get_coregistered_reference_image(
       "sample.zarr",
       reference_key="hne",
       channel_index=0,
       mask_low=False,
   )

   above = get_coregistered_msi_mask_image(
       "sample.zarr",
       if_mask.above_mask,
       msi_dataset="nanoDESI Negative",
       reference_key="hne",
   )

   fig, ax = plt.subplots(figsize=(8, 8))
   ax.imshow(ref.data, cmap="gray", interpolation="none")
   ax.imshow(above.data, cmap="autumn", alpha=0.45, interpolation="none")
   ax.set_axis_off()
   fig.savefig("if_threshold_overlay.png", dpi=300, bbox_inches="tight")

For publication figures with one or more ion overlays, colorbars, scale bars,
and optional editable PDF output:

.. code-block:: python

   from viu_chem.coreg_figures import export_reference_ion_overlay

   export_reference_ion_overlay(
       "sample.zarr",
       reference_key="hne",
       ion_layers=[
           {
               "mz": 611.1447,
               "msi_dataset": "nanoDESI Negative",
               "label": "GSSG",
               "cmap": "viridis",
           },
           {
               "mz": 306.0765,
               "msi_dataset": "nanoDESI Negative",
               "label": "GSH",
               "cmap": "magma",
           },
       ],
       scale_bar_length=1,
       scale_bar_units="mm",
       output_path="sample_hne_ion_overlay.tif",
       editable_output_path="sample_hne_ion_overlay.pdf",
       dpi=600,
   )

For treatment/replicate grids, use ``coregistered_campaign_drawgrid``.

Coregistration API Reference
----------------------------

These are the public functions and data containers used by the workflow above.
The full module listings are also available in :mod:`viu_chem.msi_coregistration`,
:mod:`viu_chem.coreg_figures`, and :mod:`viu_chem.coreg_gui`.

Dataset Preparation and Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: viu_chem.msi_coregistration

.. autosummary::

   convert_input_to_zarr
   prepare_coregistration_zarr
   prepare_coregistration_batch
   list_coregistration_msi_datasets
   embed_msi_dataset
   rename_msi_dataset
   delete_msi_dataset
   add_reference_image

Registration and Display Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   CoregistrationDataset
   save_coregistration
   launch_coregistration_gui
   xy_to_yx_matrix
   auto_contrast_limits
   finite_data_limits
   normalize_image_for_registration
   prepare_ion_for_display
   sitk_affine_from_fixed_to_moving_matrix
   sitk_transform_to_homogeneous_matrix

Annotation and ROI Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   import_geojson_annotations
   delete_geojson_annotations
   rescale_geojson_annotations
   transform_geojson_annotations
   create_annotation_region_mask
   summarize_annotation_region_spectra
   summarize_msi_pixel_mask_spectra

Threshold Masks and Threshold Summaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   MSIThresholdMask
   MSIThresholdSpectra
   create_msi_threshold_mask
   summarize_msi_threshold_spectra
   ReferenceThresholdMask
   ReferenceThresholdSpectra
   sample_reference_channel_values_at_msi_pixels
   summarize_reference_channels_in_msi_mask
   create_reference_threshold_mask
   summarize_reference_threshold_spectra

Threshold Annotations
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   create_msi_threshold_annotation
   create_pooled_msi_threshold_annotation
   create_reference_threshold_annotation

Colocalization and Correlation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   colocalized_msi_features
   colocalized_msi_features_between_datasets
   correlate_msi_features_with_reference_channels

Coregistered Figure API
~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: viu_chem.coreg_figures

.. autosummary::

   CoregisteredImage
   mask_low_intensity_pixels
   get_coregistered_reference_image
   get_coregistered_ion_image
   get_coregistered_msi_mask_image
   get_coregistered_image_layers
   reference_rgb_composite
   export_reference_ion_overlay
   coregistered_campaign_drawgrid

GUI Entry Point
~~~~~~~~~~~~~~~

.. currentmodule:: viu_chem.coreg_gui

.. autosummary::

   launch_coregistration_gui
