Quick Start Guide
=================

``viu_chem`` is a small collection of helper functions for mass spectrometry
and MSI workflows. The package is organized by task: chromatogram utilities in
``chem412``, plotting helpers in ``Figures``, MSI image processing in
``MSI_Process``, and MSI statistics/coregistration tools in their own modules.

Installation
------------

Install the released package from PyPI:

.. code-block:: bash

   pip install viu-chem

For development, clone the repository and install it locally:

.. code-block:: bash

   git clone https://github.com/<your-org-or-user>/viu_chem.git
   cd viu_chem
   poetry install

Optional GUI/coregistration dependencies are heavier than the core scientific
helpers. Install them only when you need napari/SpatialData workflows:

.. code-block:: bash

   pip install "viu-chem[coregistration]"

Basic Imports
-------------

Import the submodule you need for the workflow you are running:

.. code-block:: python

   from viu_chem import chem412
   from viu_chem import Figures
   from viu_chem import MSI_Process

Extract a Chromatogram
----------------------

Use ``chem412.extract_data`` to extract one or more m/z traces from an ``mzML``
file. The function returns one dataframe-like array per scan filter when the
file contains multiple filters.

.. code-block:: python

   from viu_chem import chem412

   mzml_path = "data/sample.mzML"
   target_mz = 104.1070

   traces = chem412.extract_data(
       mzml_path,
       mz_list=[target_mz],
       tol_mode="ppm",
       tol=10,
   )

   print(traces.keys())

Plot a Calibration Curve
------------------------

Use ``Figures.cal_curve`` when you already have concentration and response
values.

.. code-block:: python

   import matplotlib.pyplot as plt
   from viu_chem import Figures

   concentrations = [0.5, 1, 5, 10, 25]
   peak_areas = [1020, 2150, 10400, 20800, 51400]

   fig, ax = plt.subplots()
   ax, coeffs, r2 = Figures.cal_curve(
       concentrations,
       peak_areas,
       ax=ax,
       xlabel="Concentration (uM)",
       ylabel="Peak area",
   )
   fig.savefig("calibration_curve.png", dpi=300, bbox_inches="tight")

Generate an MSI Ion Image
-------------------------

For a single ``imzML`` file, retrieve an ion image and save it with
``MSI_Process``:

.. code-block:: python

   from viu_chem import MSI_Process

   imzml_path = "data/sample.imzML"
   mz = 104.1070

   ion_image = MSI_Process.get_image_matrix(imzml_path, mz=mz, tol=10)
   aspect = MSI_Process.get_aspect_ratio(imzml_path)

   MSI_Process.draw_ion_image(
       ion_image,
       cmap="viridis",
       mode="save",
       path="sample_104_1070.tif",
       asp=aspect,
       cut_offs=(5, 95),
   )

Where to Go Next
----------------

See :doc:`ExampleMSI` for a command-line MSI image export workflow, and
:doc:`ExampleChem412` for chromatogram overlay, peak integration, and
calibration curve plotting. See :doc:`coregistration` for SpatialData zarr
preparation, napari registration, annotation masks, threshold summaries,
colocalization, and coregistered figure export.
