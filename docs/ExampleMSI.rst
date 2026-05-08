Example - MSI Image Export Workflow
===================================

This example shows how to generate one or more MSI ion images from the command
line. It uses the ``MSI_Process`` module to read an ``imzML`` file, extract
ion images within a ppm tolerance, optionally TIC-normalize them, and save the
result as image files.

Single Image Script
-------------------

Create a file named ``export_ion_image.py``:

.. code-block:: python

   import argparse
   from pathlib import Path
   import numpy as np

   from viu_chem import MSI_Process


   def main():
       parser = argparse.ArgumentParser(
           description="Export a single ion image from an imzML file."
       )
       parser.add_argument("imzml", help="Input imzML file")
       parser.add_argument("mz", type=float, help="Target m/z")
       parser.add_argument("output", help="Output image path, usually .tif or .png")
       parser.add_argument("--tol", type=float, default=10, help="Tolerance in ppm")
       parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap")
       parser.add_argument(
           "--tic-normalize",
           action="store_true",
           help="Normalize ion image by the TIC image before saving",
       )
       parser.add_argument(
           "--cutoffs",
           nargs=2,
           type=float,
           default=(5, 95),
           metavar=("LOW", "HIGH"),
           help="Lower and upper percentile cutoffs",
       )
       args = parser.parse_args()

       imzml_path = Path(args.imzml)
       ion_image = MSI_Process.get_image_matrix(
           str(imzml_path),
           mz=args.mz,
           tol=args.tol,
       )

       if args.tic_normalize:
           tic_image = MSI_Process.get_TIC_image(str(imzml_path))
           ion_image = np.divide(
               ion_image,
               tic_image,
               out=np.zeros_like(ion_image),
               where=tic_image != 0,
           )

       aspect = MSI_Process.get_aspect_ratio(str(imzml_path))
       MSI_Process.draw_ion_image(
           ion_image,
           cmap=args.cmap,
           mode="save",
           path=args.output,
           cut_offs=tuple(args.cutoffs),
           asp=aspect,
       )


   if __name__ == "__main__":
       main()

Run it from the terminal:

.. code-block:: bash

   python export_ion_image.py data/sample.imzML 104.1070 outputs/sample_104_1070.tif --tol 10 --tic-normalize

The output image is percentile-scaled using ``--cutoffs`` and drawn with the
physical aspect ratio stored in the ``imzML`` metadata.

Batch Export from a Folder
--------------------------

For a folder of experiments, ``MSI_Process.bulk_image_export`` can export the
same target list across many ``imzML`` files. This expects each experiment to
live in its own subfolder and uses ``search_pattern`` to find the desired
``imzML`` file inside each folder.

.. code-block:: python

   from viu_chem import MSI_Process

   MSI_Process.bulk_image_export(
       dir="data/msi_campaign",
       search_pattern=".imzML",
       save_path="outputs/msi_images",
       mz_list=[104.1070, 184.0733, 760.5851],
       target_list=["target_104", "target_184", "target_760"],
       tolerance=10,
       uniform_scale=True,
       smooth=False,
   )

This creates one output folder per target under ``outputs/msi_images/images``.
Use ``include_codes`` when you only want to export a subset of samples:

.. code-block:: python

   MSI_Process.bulk_image_export(
       dir="data/msi_campaign",
       search_pattern=".imzML",
       save_path="outputs/msi_images",
       mz_list=[104.1070],
       target_list=["target_104"],
       include_codes=["control", "treated"],
       tolerance=10,
   )

Notes
-----

``tol`` and ``tolerance`` are interpreted as ppm values in these workflows.
For publication figures, inspect the output scaling and consider exporting
with ``uniform_scale=True`` when comparing the same ion across samples.
