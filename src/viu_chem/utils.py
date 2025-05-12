import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

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