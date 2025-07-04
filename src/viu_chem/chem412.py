import imzml_writer.utils as utils
import pymzml
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
import json
import numpy as np

def convert_to_mzML(path:str, file_type:str):
    try:
        utils.RAW_to_mzML(path,blocking=True)
        utils.clean_raw_files(path, file_type)
        return True
    except Exception as e:
        return e




def extract_from_existing_run(run, mz_list:list, tol_mode:str='ppm', tol:float = 10):
    data = {mz:[] for mz in mz_list}
    for spectrum in run:
        for mz_idx, mz_search in enumerate(mz_list):
            if tol_mode == "unit":
                for idx, mz in enumerate(spectrum.mz):
                    if (float(mz) > (mz_search - 0.5) and float(mz) < (mz_search + 0.5)):
                        data[mz_search].append(float(spectrum.i[idx]))

            elif tol_mode == "ppm":
                summed_signal = 0
                for idx, mz in enumerate(spectrum.mz):
                    if (float(mz) > (mz_search - (mz_search*tol/1e6))) and (float(mz) < (mz_search + (mz_search*tol/1e6))):
                        summed_signal+=float(spectrum.i[idx])

                data[mz_search].append(summed_signal)
    
    return data
        

def extract_data(path,mz_list, tol_mode:str='ppm', tol:float = 10):
    run = pymzml.run.Reader(path)
        
    filt_strings = []
    data = {}
    for idx, spectrum in enumerate(run):
        filt_strings.append(spectrum["filter string"])
        length = idx
        
    unique_filts = list(set(filt_strings)) #Gets unique filters
    

    #initialize dictionary
    for filt in unique_filts:
        data[filt]=np.zeros((length+1, len(mz_list)+1))
                
    for filt in unique_filts:
        filt_idx = -1
        for spectrum in run:
            if spectrum["filter string"] == filt:
                filt_idx +=1
                data[filt][filt_idx, 0] = spectrum.scan_time_in_minutes()

                for mz_idx, mz_search in enumerate(mz_list):
                    
                    if tol_mode == "unit":
                        for idx, mz in enumerate(spectrum.mz):
                            if (float(mz) > (mz_search - 0.5) and float(mz) < (mz_search + 0.5)):
                                data[filt][filt_idx, mz_idx+1] = float(spectrum.i[idx])

                    elif tol_mode == "ppm":
                        for idx, mz in enumerate(spectrum.mz):
                            if (float(mz) > (mz_search - (mz_search*tol/1e6))) and (float(mz) < (mz_search + (mz_search*tol/1e6))):
                                data[filt][filt_idx, mz_idx+1] = float(spectrum.i[idx])


    if None in data.keys() and len(data.keys()) == 1:
        return data[None]
    else:
        return data

def extract_peak(data:np.array, index:int, ret_window:tuple):
    within_window = data[(data[:,0] > ret_window[0]) & (data[:,0] < ret_window[1]), index]
    baseline_subtract = within_window - np.min(within_window)
    sum_method = np.sum(baseline_subtract)

    return sum_method


def calcurve(x, y, xlabel:str="Your x label here!", ylabel:str="Your y label here!", color:str="#8C4FA4"):
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)

    #Generate linear fit
    coeffs = np.polyfit(x, y, 1)
    poly_eq = np.poly1d(coeffs)
    x_fit = np.unique(x)
    y_fit = poly_eq(x_fit)

    y_pred = poly_eq(x)
    ss_res = np.sum((y - y_pred) ** 2)  # Residual sum of squares
    ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
    r2 = 1 - (ss_res / ss_tot)
    
    plt.plot(x_fit, y_fit, color=color,linestyle='--')
    plt.scatter(x, y, marker='s',edgecolors='k', color=color)

    # Prepare the equation string
    slope, intercept = coeffs
    equation = f"y = {slope:.2f}x + {intercept:.2f}\nR² = {r2:.3f}"

    # Add text annotation for the equation
    plt.text(0.2, 0.95, equation, transform=plt.gca().transAxes, va='top',ha='center', color=color)
    # plt.text(0.05, 0.90, f"R² = {r2:.2f}", transform=plt.gca().transAxes, verticalalignment='top', color=color)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

    

