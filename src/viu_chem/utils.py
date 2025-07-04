import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
import viu_chem.MSI_Process as msi
import viu_chem.chem412
import pymzml
import re

def cv_to_csv(directory:str,mz_list:list,tolerance:float=10):
    """Converts a directory of .raw files for a FAIMS CV Scan to mzML (if needed), then outputs a dataframe with each mz intensity
    against the CV value (from scan filter) as a CSV (one for each raw file in the directory)
    
    :param directory: Folder containing .raw files
    :param mz_list: list of mz to export a CV spectrum for
    :param tolerance: Tolerance to search for each mz (in ppm)"""

    cv_regex = r"(-?\d+(?:\.\d+)?)"

    directory_files = os.listdir(directory)
    if "Output mzML Files" not in directory_files:
        msi.convert_from_RAW(os.path.abspath(directory),stop_at_mzML=True)

    
    active_path = os.path.join(directory, "Output mzML Files")
    mzML_files = [file for file in os.listdir(active_path) if file.endswith(".mzML")]

    os.makedirs(os.path.join(directory,"Output CSV Files"),exist_ok=True)
    for file in mzML_files:
        cv_vals = []
        run = pymzml.run.Reader(os.path.join(active_path,file))
        for spec in run:
            mtch = re.search(cv_regex,spec['filter string'])
            if mtch:
                cv_vals.append(float(mtch.group(1)))
        
        data = viu_chem.chem412.extract_from_existing_run(run,mz_list,tol=tolerance)
        df = pd.DataFrame.from_dict(data)
        df.index = cv_vals
        save_path = os.path.join(directory,"Output CSV Files",f"{file.split(".mzML")[0]}.csv")
        df.to_csv(save_path)








def extract_calibration(datafiles:list[str], concs:list[float],mz:float,tolerance:float=25, plot:bool=True) ->dict:
    """Reads in a list of thermo-formatted .csv holding mass spectral data and a list of calibration concentrations and extracts
    the specified m/z within a tolerance (ppm). Optionally plots the figure. Returns a dictionary containing x & y scatter data (including intensities)
    and calibration data
    
    :param datafiles: List of datafiles with full or relative pathing
    :param concs: List of calibration concentrations, specified as floats
    :param mz: m/z value to target
    :param tolerance: Tolerance window within which to search for the m/z (max in window; default 25ppm)
    :param plot: bool specifying whether or not to plot the resulting figure"""

    #Compute search bounds for mz tolerance
    low_bound = mz - (mz * tolerance / 1e6)
    up_bound = mz + (mz * tolerance / 1e6)

    ##Load in data files
    data_list = []
    for file in datafiles:
        data_list.append(pd.read_csv(file,header=7))

    #Initialize figure
    if plot:
        (fig, axes) = plt.subplots(1,2)
        fig.set_size_inches(8, 5)


    ##Extract data from each file (max within tolerance window) and plot the mass spectrum about that tolerance
    intensities = []
    for idx, data in enumerate(reversed(data_list)):
        if plot:
            label_string = f"{concs[-(idx+1)]} µM"
            axes[0].fill_between(data.Mass,data.Intensity,label=label_string,alpha=0.3)
        local_data = data[(data.Mass > low_bound) & (data.Mass < up_bound)]
        intensities.append(local_data.Intensity.max())

    if plot:
        ##Stylize and format the mass spectrum
        axes[0].set_xlim([low_bound, up_bound])
        axes[0].set_ylim([0, 1.3*np.max(intensities)])
        axes[0].set_xlabel("m/z")
        axes[0].set_ylabel("Intensity")
        axes[0].legend()

        ##Plot/stylize the calibration points
        axes[1].scatter(concs[::-1], intensities, marker='s', edgecolors='k')
        axes[1].set_xlim([0, 1.1*np.max(concs)])
        axes[1].set_ylim([0, 1.3*np.max(intensities)])
        axes[1].set_xlabel("Concentration (µM)")
        axes[1].set_ylabel("Intensity")

    #Fit / draw the calibration curve
    slope, intercept, r_value, p_value, std_err = stats.linregress(concs[::-1], intensities)
    r_value = r_value ** 2
    if plot:
        y_fit = (slope * np.array(concs[::-1])) + intercept
        axes[1].plot(concs[::-1], y_fit)
        sign = None
        if intercept < 0:
            sign = "-"
        elif intercept > 0:
            sign = "+"

        if sign:
            eqn_string = f"y = {slope:.2e} * x {sign} {abs(intercept):.2e}\nR^2 = {r_value:.3f}"
        else:
            eqn_string = f"y = {slope:.2e} * x \n R^2 = {r_value:.3f}"

        plt.annotate(eqn_string, xy=(0.4, 0.75), xycoords='axes fraction', verticalalignment='center', ha='center')

        plt.show()
    
    return_dictionary = {
        "data":{
            "concs": concs,
            "intensities": intensities[::-1]
        },
        "model": {
            "slope": slope,
            "intercept": intercept,
            "r2": r_value,
            "p_val": p_value,
            "std_err": std_err
        }
    }
    return return_dictionary