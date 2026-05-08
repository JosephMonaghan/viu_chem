Example - Chromatograms and Calibration Curves
==============================================

This workflow uses ``chem412`` to extract chromatograms from ``mzML`` files,
overlay several traces, integrate a peak, and plot a calibration curve with
``Figures.cal_curve``.

Find the Scan Filter
--------------------

Many vendor-converted ``mzML`` files contain multiple scan filters. Start by
printing the available filters, then choose the one that corresponds to the
MS1 trace you want.

.. code-block:: python

   from viu_chem import chem412

   mzml_path = "data/calibration/std_10uM.mzML"
   filters = chem412.get_scan_filters(mzml_path)

   for filt in filters:
       print(filt)

Extract and Overlay Chromatograms
---------------------------------

The example below extracts the same m/z from several calibration files and
plots each chromatogram on the same axes.

.. code-block:: python

   from pathlib import Path
   import matplotlib.pyplot as plt
   import pandas as pd

   from viu_chem import chem412

   target_mz = 104.1070
   ms1_filter = "FTMS + p ESI Full ms [70.0000-1000.0000]"

   calibration_files = {
       0.5: Path("data/calibration/std_0p5uM.mzML"),
       1.0: Path("data/calibration/std_1uM.mzML"),
       5.0: Path("data/calibration/std_5uM.mzML"),
       10.0: Path("data/calibration/std_10uM.mzML"),
       25.0: Path("data/calibration/std_25uM.mzML"),
   }

   fig, ax = plt.subplots(figsize=(7, 4))
   integrated_rows = []

   for concentration, path in calibration_files.items():
       extracted = chem412.extract_data(
           path,
           mz_list=[target_mz],
           tol_mode="ppm",
           tol=10,
       )

       trace = extracted[ms1_filter]
       time = trace[:, 0]
       signal = trace[:, 1]

       ax.plot(time, signal, label=f"{concentration:g} uM")

       peak = chem412.integrate_peak(
           time,
           signal,
           peak_x=5.35,
           times=(5.05, 5.70),
           plot=False,
       )
       integrated_rows.append(
           {
               "concentration": concentration,
               "area": peak["area"],
               "peak_time": peak["peak_time"],
               "start_time": peak["start_time"],
               "end_time": peak["end_time"],
           }
       )

   ax.set_xlabel("Retention time (min)")
   ax.set_ylabel("Signal")
   ax.legend(title="Standard")
   fig.tight_layout()
   fig.savefig("overlaid_chromatograms.png", dpi=300)

   calibration_df = pd.DataFrame(integrated_rows).sort_values("concentration")
   calibration_df.to_csv("calibration_integrations.csv", index=False)

``peak_x`` tells ``integrate_peak`` which peak to integrate. The optional
``times`` argument manually fixes the start and end of integration, which is
useful when you want consistent bounds across a calibration series.

Plot the Calibration Curve
--------------------------

Once peak areas are collected, pass the concentrations and areas to
``Figures.cal_curve``.

.. code-block:: python

   import matplotlib.pyplot as plt
   from viu_chem import Figures

   fig, ax = plt.subplots(figsize=(4.5, 4))
   ax, coeffs, r2 = Figures.cal_curve(
       calibration_df["concentration"].to_numpy(),
       calibration_df["area"].to_numpy(),
       ax=ax,
       xlabel="Concentration (uM)",
       ylabel="Integrated peak area",
       color="#8C4FA4",
   )

   fig.tight_layout()
   fig.savefig("calibration_curve.png", dpi=300)

   slope, intercept = coeffs
   print(f"slope: {slope:.4f}")
   print(f"intercept: {intercept:.4f}")
   print(f"R2: {r2:.4f}")

Putting It Together
-------------------

For a reusable command-line script, combine the extraction, integration, and
plotting steps and expose the key values as arguments:

.. code-block:: bash

   python calibration_workflow.py \
       --mz 104.1070 \
       --filter "FTMS + p ESI Full ms [70.0000-1000.0000]" \
       --peak-x 5.35 \
       --start 5.05 \
       --end 5.70 \
       --output-dir outputs/calibration

The exact scan filter string, target m/z, and integration window should be
updated for the instrument method and compound being quantified.
