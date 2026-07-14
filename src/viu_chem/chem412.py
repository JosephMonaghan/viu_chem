import imzml_writer.utils as utils
import pymzml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from pathlib import Path
from scipy.signal import find_peaks, savgol_filter
from pathlib import Path


def convert_to_mzML(path:str | Path, file_type:str=".raw"):
    """Converts a directory of raw MS files into mzML using imzML_Writer's utilities (via pwiz / Docker image)
    
    :param path: Path to target directory containing raw vendor files
    :param file_type: File extension for files (e.g. '.raw' for Thermo)"""
    if isinstance(path,Path):
        path = str(path.absolute())
    try:
        utils.RAW_to_mzML(path)
        utils.clean_raw_files(path, file_type)
        return True
    except Exception as e:
        return e
        

def extract_spectra(path:str | Path, window:tuple[float,float],filter_str:str) -> list[np.array]:
    """Extracts unaligned spectra from an mzML within the specified window and scan filter
    
    :param path: Path to the target mzML
    :param window: Tuple of form (start, end) specifying where to extract spectra from
    :param filter_str: Filter string from which to extract the spectra from
    """
    run = pymzml.run.Reader(path)

    scans = []
    for idx, spec in enumerate(run):
        scan_time = spec.scan_time_in_minutes()
        if scan_time > window[0] and scan_time < window[1]:
            if spec['filter string'] == filter_str:
                new_scan = np.column_stack([spec.mz, spec.i])
                scans.append(new_scan)
    
    return scans

def get_scan_filters(path:str | Path) -> list[str]:
    """Scans through an mzML and returns a list of available scan filters
    
    :param path: Path to target mzML"""
    run = pymzml.run.Reader(path)
    filters = []
    for spec in run:
        if spec['filter string'] not in filters:
            filters.append(spec['filter string'])

    return filters

def extract_cv_spectrum(path:Path, mz: float | list[float], tol: float = 10) -> pd.DataFrame:
    """Extracts the CV spectrum from a FAIMS scan acquired from the tune page. Only accomodates one base scan filter.
    
    :param path: Path to the source mzML
    :param mz: m/z value or list of values to extract
    :param tol: m/z tolerance to extract at
    
    :return: DataFrame with a column for each m/z and the cv as the index"""
    run = pymzml.run.Reader(path)
    if isinstance(mz, float):
        mz = [mz]
    
    data = []
    
    for spec in run:
        local_dict = {}
        local_dict['cv'] = float(spec['FAIMS compensation voltage'])
        for local_mz in mz:
            low, high = utils.calculate_tolerance_window(local_mz, tol)
            intensity = spec.i[(spec.mz > low) &  (spec.mz < high)]
            if len(intensity) > 0:
                intensity = np.sum(intensity)
            else:
                intensity = 0
            
            local_dict[local_mz] = intensity
        
        data.append(local_dict)
            
    run.close()
    df = pd.DataFrame(data)
    df = df.set_index('cv')
    return df


def get_ms2(path: str | Path, precursor_mz:float, window:tuple[float, float],tol:float=0.5) -> list[np.array]:
    """Finds ms2 spectra (unaligned) within the window for the specified m/z in the target mzML. Tolerant to some float in precursor m/z from DDA.
    
    :param path: Path to the target mzML
    :param precursor_mz: Target m/z
    :window: Tuple of form (start, end) specifying the window in which to retrieve spectra
    :tol: Tolerance window in which spectra can be extracted (default = 0.5)"""
    run = pymzml.run.Reader(path)

    spectra = []
    for spec in run:
        scan_time = spec.scan_time_in_minutes()
        if scan_time > window[0] and scan_time<window[1]:                
            if spec.ms_level >= 2:
                selected_prec = float(spec['isolation window target m/z'])
                if selected_prec > (precursor_mz-tol) and selected_prec < (precursor_mz+tol):
                    spectra.append(np.column_stack([spec.mz, spec.i]))
                
    return spectra


def extract_data(path:str | Path,mz_list:float | list[float], tol_mode:str='ppm', tol:float = 10, ms_level:list[int]=[1]) -> dict:
    """Extracts chromatograms from the target mzML across all scan filters for the specified m/z list.
    
    :param path: Path to target mzML
    :param mz_list: List of target m/z or single target (float)
    :param tol_mode: Specify tolerance mode as either 'unit' or 'ppm'
    :param tol: Tolerance for high resolution extraction (default 10 ppm)
    :param ms_level: MS level filter, ignores MS2 by default for DDA compatibility
    
    :return: Dictionary containing a pd.Dataframe of form time, mz_1, mz_2, ... mz_n for each scan filter"""
    run = pymzml.run.Reader(path)

    if isinstance(mz_list, float):
        mz_list = [mz_list]
        
    filt_strings = []
    data = {}
    for idx, spectrum in enumerate(run):
        if spectrum.ms_level in ms_level:
            filt_strings.append(spectrum["filter string"])
        length = idx
        
    unique_filts = list(set(filt_strings)) #Gets unique filters
    

    #initialize dictionary
    for filt in unique_filts:
        data[filt]=np.zeros((length+1, len(mz_list)+1))
                
    for filt in unique_filts:
        filt_idx = -1
        for spectrum in run:
            mzs = np.array(spectrum.mz)
            intensities = np.array(spectrum.i)


            if spectrum["filter string"] == filt:
                filt_idx +=1
                data[filt][filt_idx, 0] = spectrum.scan_time_in_minutes()

                for mz_idx, mz_search in enumerate(mz_list):
                    
                    if tol_mode == "unit":
                        for idx, mz in enumerate(spectrum.mz):
                            if (float(mz) > (mz_search - 0.5) and float(mz) < (mz_search + 0.5)):
                                data[filt][filt_idx, mz_idx+1] = float(spectrum.i[idx])

                    elif tol_mode == "ppm":
                        low = mz_search - (mz_search*tol/1e6)
                        high = mz_search + (mz_search*tol/1e6)

                        matches = intensities[(mzs > low) & (mzs < high)]
                        if len(matches) > 0:
                            data[filt][filt_idx, mz_idx+1] = np.max(matches)


    for filt, data_vals in data.items():
        if data_vals[0,0] == 0:
            first_line = data_vals[0,:]
            non_zeros = data_vals[data_vals[:,0] != 0,:]
            data[filt] = np.vstack((first_line, non_zeros))
        else:
            data[filt] = data_vals[data_vals[:,0] != 0,:]

    if None in data.keys() and len(data.keys()) == 1:
        return data[None]
    else:
        return data

def extract_7010(path:Path | str) -> pd.DataFrame:
    """Extracts data from an Agilent .csv file export.
    
    :param path: path to target csv
    :return: pd.Dataframe of form time, mz_1, mz_2, ... mz_n in the csv output"""
    data = {}
    row_header = None
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == "#Point":
                continue
            try:
                float(row[0])
                times.append(float(row[1]))
                signals.append(float(row[2]))
            except ValueError:
                if row_header != None:
                    data[row_header] = {"times":times, "sig":signals}
                row_header = row[0]
                times = []
                signals = []
    
    data[row_header] = {"times": times,"sig": signals}
    
    new = True
    for key, val in data.items():
        if new:
            df = pd.DataFrame(data=val)
            df.columns = ["Time (mins)", key]
            new = False
        
        df[key] = val['sig']
    
    return df


def average_centroided_spectra(spectra:list[np.array],mz_tolerance:float=0.001) -> np.array:
    """Averages a small number of spectra (<1000) to a coherent mass frame
    
    :param spectra: list of unaligned spectra to align
    :param mz_tolerance: Minimum spacing between peaks
    
    :return: np.array of the coherent mz (first column) and intensity (2nd column)"""

    if len(spectra) < 1:
        return
    
    all_mz = np.concatenate([spec[:,0] for spec in spectra])
    all_intensities = np.concatenate([spec[:,1] for spec in spectra])

    order = np.argsort(all_mz)
    all_mz = all_mz[order]
    all_intensities = all_intensities[order]

    pooled_mz = []
    pooled_intensity = []

    start = 0
    n = len(all_mz)

    while start < n:
        end = start + 1
        center = all_mz[start]

        while end < n and abs(all_mz[end] - center) <= mz_tolerance:
            end += 1

        mz_group = all_mz[start:end]
        int_group = all_intensities[start:end]

        # intensity-weighted center gives a better pooled location
        if np.sum(int_group) > 0:
            group_mz = np.sum(mz_group * int_group) / np.sum(int_group)
        else:
            group_mz = np.mean(mz_group)

        group_intensity = np.sum(int_group)

        pooled_mz.append(group_mz)
        pooled_intensity.append(group_intensity)

        start = end
    
    pooled_mz = np.asarray(pooled_mz)
    pooled_intensity = np.asarray(pooled_intensity)

    peak_idx, _ = find_peaks(
        pooled_intensity,
        prominence=2.5,
    )

    consensus_mz = pooled_mz[peak_idx]
    aligned_spec = np.zeros([len(spectra), len(consensus_mz)])
    for i, spec in enumerate(spectra):
        spec_mz = spec[:,0]
        spec_i = spec[:,1]
                
        # sorted assumption helps nearest-neighbor lookup
        spec_order = np.argsort(spec_mz)
        spec_mz = spec_mz[spec_order]
        spec_i = spec_i[spec_order]

        for j, target_mz in enumerate(consensus_mz):
            pos = np.searchsorted(spec_mz, target_mz)

            candidates = []
            if pos < len(spec_mz):
                candidates.append(pos)
            if pos > 0:
                candidates.append(pos - 1)

            best_intensity = 0.0
            best_delta = np.inf

            for k in candidates:
                delta = abs(spec_mz[k] - target_mz)
                if delta <= mz_tolerance and delta < best_delta:
                    best_delta = delta
                    best_intensity = spec_i[k]

            aligned_spec[i, j] = best_intensity


    mean_intensity = aligned_spec.mean(axis=0)
    return np.column_stack((consensus_mz, mean_intensity))


def metabolite_overview(path:str | Path, tgt_mz:float, MS1_filter:str, window:tuple[float, float]| None=None, default_window_width:float=0.3,plot:bool=True) ->tuple[pd.DataFrame, float, np.array]:
    """Utility function to quickly retrieve the chromatogram / average MS2 spectrum (if available) for the specified m/z
    
    :param path: Path to source mzML
    :param tgt_mz: m/z to extract
    :param MS1_filter: Filter for the chromatogram extraction
    :param window: Optional specification of where to extract the MS2 spectra from
    :param default_window_width: default window about the max intensity to extract MS2 from
    :param plot: Boolean on whether to draw the chromatogram / MS2 spectra for the target ion
    
    :return tgt: Chromatogram data as a pd.Dataframe of form time, int
    :return peak_time: Retention time of peak intensity (mins)
    :return avg_spec: np array of average, aligned MS2 spectrum"""
    data = extract_data(path,[tgt_mz])
    tgt = data[MS1_filter]

    if not window:
        peak_idx, _ = find_peaks(tgt[:,1], prominence = 3)
        peak_sigs = pd.DataFrame(tgt[peak_idx,:],columns=["time", "intensity"])
        print(peak_sigs)
        biggest = peak_sigs.loc[peak_sigs['intensity'].idxmax()]
        print(biggest)
        peak_time = biggest["time"]
        window = (peak_time-default_window_width, peak_time+default_window_width)
        print(window)
    
    spectra = get_ms2(path, tgt_mz,window)
    avg_spec = average_centroided_spectra(spectra)

    if plot:
        fig, ax = plt.subplots()
        ax.plot(tgt[:,0],tgt[:,1], color='k')
        ax.set_xlabel("Time (mins)")
        ax.set_ylabel("Signal")
        ax.axvspan(window[0],window[1],facecolor='red',alpha=0.3,edgecolor=None)

        ax2 = ax.inset_axes([0.6, 0.6, 0.3, 0.3])
        ax2.vlines(avg_spec[:,0],0, avg_spec[:,1],color='red')
        ax2.axhline(0,color='k')
        ax2.set_ylim(0, np.max(avg_spec[:,1]*1.3))
        ax2.set_xlabel("m/z",style='italic')
        ax2.set_ylabel("Intensity")
    
    return tgt, peak_time, avg_spec


def integrate_peak(
    x:np.array,
    y:np.array,
    peak_x:float | None=None,
    prominence:float=3,
    smooth_window:int=11,
    polyorder:int=2,
    edge_frac:float=0.02,
    max_width:float=None,
    plot:bool=False,
    times:tuple[float, float] | None = None,
) -> dict:
    """Integrates under a peak from the specified x and y data, accounting for a moving baseline over the peak.
    
    :param x: X data
    :param y: Y data
    :param peak_x: optional specification of the peak time / location in x value. Finds automatically if unspecified
    :param prominence: Required peak prominence for auto peak detection
    :param smooth_window: Smoothing width for savitzsky-golay feature on peak finding
    :param polyorder: Order of savitzsky-golay fit
    :edge_frac: Threshold relative height over baseline to cut off integration
    :max_width: Maximum width of peak integration window for autodetection
    :times: Manual specification of start and end locations for integration (optional; tuple[start, end])"""

    x = np.asarray(x)
    y = np.asarray(y)

    
    y_smooth = savgol_filter(y, smooth_window, polyorder)
    
    # Determine peak apex index
    if peak_x is not None:
        peak_idx = int(np.argmin(np.abs(x - peak_x)))
    else:
        peaks, props = find_peaks(y_smooth, prominence=prominence)
        if len(peaks) == 0:
            raise ValueError("No peaks found.")
        peak_idx = peaks[np.argmax(props["prominences"])]


    # Crude valley estimate on each side
    left_min_idx = np.argmin(y_smooth[:peak_idx]) if peak_idx > 0 else 0
    right_min_idx = (
        peak_idx + np.argmin(y_smooth[peak_idx:])
        if peak_idx < len(y_smooth) - 1
        else len(y_smooth) - 1
    )

    # Initial baseline estimate from crude valleys
    x1_init, x2_init = x[left_min_idx], x[right_min_idx]
    y1_init, y2_init = y_smooth[left_min_idx], y_smooth[right_min_idx]

    if x2_init == x1_init:
        baseline_at_apex = y1_init
    else:
        baseline_at_apex = y1_init + (y2_init - y1_init) * (x[peak_idx] - x1_init) / (x2_init - x1_init)

    apex_y = y_smooth[peak_idx]
    peak_height = apex_y - baseline_at_apex
    threshold = baseline_at_apex + edge_frac * peak_height

    # Walk left
    left_idx = peak_idx
    steps = 0
    while left_idx > 0 and y_smooth[left_idx] > threshold:
        left_idx -= 1
        steps += 1
        if max_width is not None and steps >= max_width:
            break

    # Walk right
    right_idx = peak_idx
    steps = 0
    while right_idx < len(y_smooth) - 1 and y_smooth[right_idx] > threshold:
        right_idx += 1
        steps += 1
        if max_width is not None and steps >= max_width:
            break
    
    if times:
        left_idx = int(np.argmin(np.abs(x-times[0])))
        right_idx = int(np.argmin(np.abs(x-times[1])))


    # Final baseline between chosen edges
    x1, x2 = x[left_idx], x[right_idx]
    y1, y2 = y[left_idx], y[right_idx]

    if x2 == x1:
        baseline_y = np.full(right_idx - left_idx + 1, y1)
    else:
        baseline_y = y1 + (y2 - y1) * (x[left_idx:right_idx + 1] - x1) / (x2 - x1)

    x_seg = x[left_idx:right_idx + 1]
    y_seg = y[left_idx:right_idx + 1]
    y_corr = y_seg - baseline_y

    area = np.trapezoid(y_corr, x_seg)

    result = {
        "peak_idx": peak_idx,
        "peak_time": x[peak_idx],
        "start_idx": left_idx,
        "end_idx": right_idx,
        "start_time": x[left_idx],
        "end_time": x[right_idx],
        "area": area,
        "baseline_y": baseline_y,
    }

    if plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, label="raw")
        ax.fill_between(
            x_seg,
            y_seg,
            baseline_y,
            where=(y_seg >= baseline_y),
            alpha=0.4,
            label="integrated area",
        )
        ax.legend()
        ax.set_xlabel("Retention time")
        ax.set_ylabel("Signal")
        plt.show()

    return result

        
    
