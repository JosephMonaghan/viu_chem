import numpy as np
import os
from pyimzml.ImzMLParser import ImzMLParser
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats.mstats import pearsonr
from skimage.exposure import rescale_intensity
from scipy import spatial
from skimage.filters import threshold_otsu, threshold_yen, threshold_isodata, threshold_mean, threshold_minimum, threshold_triangle


def booltoint(v):
    """Converts common string boolean values to integer flags.
    
    :param v: String boolean value
    :return: Integer flag, or None if the value is not recognized"""
    if v == "false" or v == "False" or v == "0":
        return 0
    elif v == "true" or v == "True" or v == "1":
        return 1


def apply_threshold(img, thresh_algorithm):
    """Calculates an image threshold using a named thresholding algorithm.
    
    :param img: Image array to threshold
    :param thresh_algorithm: Name of threshold algorithm to use
    :return: Calculated threshold value"""
    if thresh_algorithm == 'otsu':
        thresh = threshold_otsu(img)
    elif thresh_algorithm == 'yen':
        thresh = threshold_yen(img)
    elif thresh_algorithm == 'isodata':
        thresh = threshold_isodata(img)
    elif thresh_algorithm == 'mean':
        thresh = threshold_mean(img)
    elif thresh_algorithm == 'minimum':
        thresh = threshold_minimum(img)
    elif thresh_algorithm == 'triangle':
        thresh = threshold_triangle(img)
    return thresh


def intensities_generator(imzmlParser, mz_index, selection=slice(None)):
    """Generates aligned intensity arrays for each spectrum in an imzML parser.
    
    :param imzmlParser: imzML parser object
    :param mz_index: Shared m/z index used for output spectra
    :param selection: Slice selecting m/z values from each spectrum
    :return: Generator yielding full intensity arrays"""
    for i in range(len(imzmlParser.coordinates)):
        full_spec = np.zeros(len(mz_index))
        mz_spec, unique_mz = np.unique(imzmlParser.getspectrum(i)[0][selection], return_index=True)
        idx = np.in1d(mz_index, mz_spec)
        full_spec[idx] = imzmlParser.getspectrum(i)[1][selection][unique_mz]
        yield full_spec


def imzml_to_df(imzml_file_path):
    """Converts an imzML file to a dataframe indexed by pixel coordinates and dataset name.
    
    :param imzml_file_path: Path to the imzML file
    :return: Dataframe containing pixel spectra"""
    dataset_name, _ = os.path.splitext(os.path.basename(imzml_file_path))

    print('Loading', imzml_file_path)
    p = ImzMLParser(imzml_file_path, parse_lib='ElementTree')
    print('Loading done!')

    # check if all spectra have the same mz axis
    num_spectra = len(p.mzLengths)
    mz_index = np.array(p.getspectrum(0)[0])
    mz_index_length = len(mz_index)
    print('m/z consistency check ...')

    # '0' = mz values, '1' = intensities
    mz_index = np.unique(np.concatenate([p.getspectrum(i)[0] for i in range(num_spectra)]))

    if len(mz_index) != mz_index_length:
        print('WARNING: Not all spectra have the same mz values. Missing values are filled with zeros!')

    print('m/z consistency check done!')

    # DEV: use small range to test bigger datasets on little memory
    mz_selection = slice(None)  # range(100)
    # load all intensities into a single data frame
    # resulting format:
    #   1 row = 1 spectrum
    #   1 column = all intensities for 1 mz, that is all values for a single intensity image
    print('DataFrame creation ...')
    msi_frame = pd.DataFrame(intensities_generator(p, mz_index, mz_selection), columns=mz_index[mz_selection])
    print('DataFrame creation done')
    print("DataFrame size equals: %i pixels, %i mz-values" % msi_frame.shape)

    msi_frame = msi_frame.fillna(0)

    xycoordinates = np.asarray(p.coordinates)[:, [0, 1]]
    multi_index = pd.MultiIndex.from_arrays(xycoordinates.T, names=("grid_x", "grid_y"))
    msi_frame.set_index(multi_index, inplace=True)

    msi_frame["dataset"] = [dataset_name] * msi_frame.shape[0]
    msi_frame = msi_frame.set_index("dataset", append=True)

    # For some data sets a small fraction of intensities (~0.1%) have been
    # negative, this might be a numerical issue in the imzml export by bruker.
    # DEV ad-hoc fix (couldn't figure out the cause or a more reasonable fix so far)
    msi_frame[msi_frame < 0] = 0

    return msi_frame
    # print('Write DataFrame ...')
    # h5_store_path = os.path.join(out_path, dataset_name + '.h5')
    # save_name_frame = 'msi_frame_' + dataset_name
    # with pd.HDFStore(h5_store_path, complib='blosc', complevel=9) as store:
    #     store[save_name_frame] = msi_frame
    # print()
    # print('done. Script completed!')


def NormalizeData(data):
    """
    Normalizes data to a range between 0 and 1.

    :param data: data to be normalized
    :return: normalized data
    """
    if np.max(data) - np.min(data) == 0:
        return np.zeros(data.shape)
    else:
        return (data - np.min(data)) / (np.max(data) - np.min(data))


def get_feature_subset(mzs, intensities, start=700, stop=1600):
    """
    Reduces a spectrum to a defined m/z range

    :param mzs: numpy array of mz values
    :param intensities: numpy array of intensities
    :param start: start m/z value
    :param stop: end m/z value
    :return: numpy array of reduced m/z values, numpy array of reduced intensities
    """

    smallest_diff_start, smallest_diff_ind_start, val_start = find_nearest_value(start, mzs)
    smallest_diff_end, smallest_diff_ind_end, val_end = find_nearest_value(stop, mzs)
    mzs_reduced = mzs[smallest_diff_ind_start:smallest_diff_ind_end + 1]
    intensities_reduced = intensities[smallest_diff_ind_start:smallest_diff_ind_end + 1]

    return mzs_reduced, intensities_reduced


def get_mean_spectrum(p, plot=False, save=False):
    """Calculates the mean spectrum across all pixels in an imzML parser.
    
    :param p: imzML parser object
    :param plot: Whether to show the mean spectrum plot
    :param save: Whether to save m/z and intensity arrays to disk
    :return: Tuple of mean m/z and intensity lists"""
    # create dictionary with m/z value as key and values with intensities
    print("identifying m/z for all pixels...")
    mz_set = set()
    for idx, _ in enumerate(tqdm(p.coordinates)):
        mzs, _ = p.getspectrum(idx)
        mz_set.update(mzs)

    mz_dict = {k: [] for k in list(mz_set)}
    mz_dict_mean = {k: [] for k in list(mz_set)}
    print("creating dictionary with all intensities for each m/z value in the MSI dataset...")
    for idx, _ in enumerate(tqdm(p.coordinates)):
        mzs, intensities = p.getspectrum(idx)
        for id, mz in enumerate(mzs):
            mz_dict[mz].append(intensities[id])

    print("calculating mean intensity for each m/z...")
    for i in tqdm(mz_dict):
        mz_dict_mean[i] = sum(mz_dict[i]) / len(mz_dict[i])

    print("sorting dictionary...")
    mz_sum = []
    int_sum = []
    for key in sorted(mz_dict_mean):
        mz_sum.append(key)
        int_sum.append(mz_dict_mean[key])

    if plot:
        plt.plot(np.asarray(mz_sum), np.asarray(int_sum))
        plt.show()

    if save:
        print("saving numpy arrays...")
        np.save('mz_sum.npy', np.asarray(mz_sum))
        np.save('int_sum.npy', np.asarray(int_sum))

    return mz_sum, int_sum


def get_combined_dataframe_from_files(file_dir, file_list, multi_index=False, groups=False):
    """Creates one dataframe from multiple imzML files.
    
    :param file_dir: Directory containing imzML files
    :param file_list: List of imzML filenames
    :param multi_index: Whether to return a dataframe with sample, x, and y as a multi-index
    :param groups: Whether to include group names parsed from filenames
    :return: Dataframe containing sample coordinates and m/z values"""
    spectra = []
    pixel = []
    file_names = []
    group_names = []
    print("reading data files from {}".format(file_dir))
    for idx, img_fl in enumerate(tqdm(file_list)):
        # print("----------------------")
        # print(os.path.join(file_dir, img_fl))
        p = ImzMLParser(os.path.join(file_dir, img_fl))
        filename = img_fl.split('.')[0]
        file_list = [filename for i in range(len(p.coordinates))]
        file_names.extend(file_list)
        if groups:
            groupname = img_fl.split('_')[0]  # expects file named accordingly: group_sampleno.imzML
            group_list = [groupname for i in range(len(p.coordinates))]
            group_names.extend(group_list)
        for idx, (x, y, z) in enumerate(p.coordinates):
            mzs, sp = p.getspectrum(idx)
            spectra.append(sp)
            pixel.append([x, y, z])
    pixel = np.array(pixel)
    spectra = np.vstack(spectra)
    file_names = np.array(file_names)
    group_names = np.array(group_names)
    if multi_index:
        multidx = pd.MultiIndex.from_arrays([np.asarray(file_names), pixel[:, 0], pixel[:, 1]],
                                            names=('sample', 'x', 'y'))
        df = pd.DataFrame(spectra, columns=mzs, index=multidx)
    else:
        df_spec = pd.DataFrame(spectra, columns=mzs)
        if groups:
            df_dict = {'group': group_names, 'sample': file_names, 'x': pixel[:, 0], 'y': pixel[:, 1]}
        else:
            df_dict = {'sample': file_names, 'x': pixel[:, 0], 'y': pixel[:, 1]}
        df = pd.DataFrame.from_dict(df_dict)
        df = pd.concat([df, df_spec], axis=1)
    return df


def get_combined_dataframe_from_group_files(file_dir, file_list):
    """Creates one dataframe from multiple imzML files for a specific group.
    
    :param file_dir: Directory containing imzML files for a specific group
    :param file_list: List of imzML filenames
    :return: Dataframe containing group, sample, coordinates, and m/z values"""
    sample = []
    spectra = []
    pixel = []
    for idx, img_fl in enumerate(file_list):
        p = ImzMLParser(os.path.join(file_dir, img_fl))
        for idx, (x, y, z) in enumerate(p.coordinates):
            mzs, sp = p.getspectrum(idx)
            spectra.append(sp)
            #animal.append(img_fl.split('.')[0][-1])
            #animal.append((img_fl.split('.')[0]).split('_')[1])
            sample.append((img_fl.split('.')[0]))
            #sample.append((img_fl.split('.')[0]).split('_')[3])
            pixel.append([x, y, z])
    pixel = np.array(pixel)
    spectra = np.vstack(spectra)
    df_spec = pd.DataFrame(spectra, columns=mzs)
    df = pd.DataFrame.from_dict({'group': np.full((len(pixel)), os.path.basename(os.path.normpath(file_dir))),
                                 'sample': sample, 'x': pixel[:, 0], 'y': pixel[:, 1]})
    df = pd.concat([df, df_spec], axis=1)
    return df


def get_summarized_spectrum(df_spectra, method='median'):
    """Calculates a mean or median summary spectrum from MSI data.
    
    :param df_spectra: Dataframe containing x, y, and m/z columns
    :param method: Summary method, either mean or median
    :return: One-row dataframe containing the summarized spectrum"""
    df_spectra = df_spectra.iloc[:,2:]
    if method == 'mean':
        df_summarized = df_spectra.mean(axis=0, skipna=True).to_frame().T
    else:
        df_summarized = df_spectra.median(axis=0, skipna=True).to_frame().T
    return df_summarized


#@jit(nopython=True)
def find_nearest_value(val, arr):
    """Finds the closest value in an array.
    
    :param val: Target value
    :param arr: Array to search
    :return: Tuple of smallest difference, nearest index, and nearest value"""
    abs_val_arr = np.abs(arr-val)
    smallest_diff_ind = abs_val_arr.argmin()
    smallest_diff = abs_val_arr[smallest_diff_ind]
    return smallest_diff, smallest_diff_ind, arr[smallest_diff_ind]


def split_string(string, split_char, pos):
    """Splits a string at the requested occurrence of a delimiter.
    
    :param string: String to split
    :param split_char: Delimiter string
    :param pos: Number of delimiter-separated fields to keep on the left
    :return: Tuple of left and right string parts"""
    temp = string.split(split_char)
    res = split_char.join(temp[:pos]), split_char.join(temp[pos:])
    return res


def get_spectra_coords_arrays(imzML_file):
    """Extracts spectra, coordinates, and m/z values from an imzML file.
    
    :param imzML_file: Path to an imzML file
    :return: Tuple of spectra array, coordinate array, and m/z array"""
    p = ImzMLParser(imzML_file)

    spectra = []
    coords = []
    for idx, (x, y, z) in enumerate(p.coordinates):
        mzs, intensities = p.getspectrum(idx)
        mzs = mzs.astype(np.float32)
        spectra.append(intensities)
        coords.append([x, y])
    spectra = np.vstack(spectra)
    coords = np.array(coords)

    return spectra, coords, mzs


def get_dataframe_from_imzML(imzML_file, multi_index=False, get_coords=False):
    """
    Generates a pandas dataframe from an imzML file with common m/z vector

    :param imzML_file: MSI data set in imzML format
    :param multi_index: if True a pandas dataframe with multi index (x,y) is created, if False x and y are columns
    :param get_coords: if True, additionally return coordinates separately
    :return:
    """
    # p = ImzMLParser(imzML_file)
    # all mz lengths are equal
    #if all(element == p.mzLengths[0] for element in p.mzLengths):
    spectra, coords, mzs = get_spectra_coords_arrays(imzML_file)
    if multi_index:
        multidx = pd.MultiIndex.from_arrays([coords[:, 0], coords[:, 1]], names=('x', 'y'))
        df_spectra = pd.DataFrame(spectra, columns=mzs, index=multidx)
    else:
        df_spectra = pd.DataFrame(spectra, columns=mzs)
        df_spectra.insert(loc=0, column='x', value=list(coords[:, 0]))
        df_spectra.insert(loc=1, column='y', value=list(coords[:, 1]))

    if get_coords:
        return df_spectra, coords
    else:
        return df_spectra
    # # all mz lengths are not equal
    # else:
    #     mz_set = set()
    #     for idx, _ in enumerate(tqdm(p.coordinates)):
    #         mzs, _ = p.getspectrum(idx)
    #         mz_set.update(mzs)
    #     if multi_index:
    #         print("mzs:", len(list(mz_set)))
    #         df = pd.DataFrame(columns=list(mz_set))
    #         df.set_index(['x', 'y'], inplace=True)
    #         print(df)
    #         for idx, coords in enumerate(tqdm(p.coordinates)):
    #             mzs, ints = p.getspectrum(idx)
    #             row_df = pd.DataFrame(columns=mzs, data=ints)
    #             print(row_df)
    #             df = pd.concat([df, row_df])
    #             print(df)


def imzML_to_csv(imzML_file, output_file=''):
    """Exports an imzML file to a CSV dataframe of spectra indexed by coordinates.
    
    :param imzML_file: Path to the imzML file
    :param output_file: Optional output CSV path
    :return: Exported dataframe"""
    spectra, coords, mzs = get_spectra_coords_arrays(imzML_file)
    coords = [tuple(i) for i in coords.tolist()]
    df = pd.DataFrame(columns=mzs, data=spectra, index=coords)
    if output_file == '':
        output_file = imzML_file.split('.')[0] + '.csv'
    df.to_csv(output_file)
    return df


def get_mz_img(pyx, msi_df, mz):
    """Constructs an ion image for a single m/z column from an MSI dataframe.
    
    :param pyx: Output image shape as y, x
    :param msi_df: MSI dataframe indexed by x and y coordinates
    :param mz: m/z column to draw
    :return: Ion image array"""
    coords = msi_df.index.tolist()
    msi_img = np.zeros(pyx).astype('uint16')
    for x_val, y_val in coords:
        msi_img[y_val, x_val] = msi_df.loc[(x_val, y_val), mz]
    return msi_img


def get_similarity_measures(mz, pyx, msi_df, img, contrast_stretch):
    """Calculates similarity between an ion image and a reference image.
    
    :param mz: m/z column to compare
    :param pyx: Output image shape as y, x
    :param msi_df: MSI dataframe indexed by x and y coordinates
    :param img: Flattened reference image
    :param contrast_stretch: Whether to contrast-stretch the ion image before comparison
    :return: Tuple of Pearson correlation and cosine similarity"""
    mz_img = get_mz_img(pyx, msi_df, mz)
    mz_img = np.nan_to_num(mz_img)
    if contrast_stretch:
        p0, p99 = np.percentile(mz_img, (0, 99.9))
        mz_img = rescale_intensity(mz_img, in_range=(p0, p99))
        #mz_img = gaussian_filter(mz_img, sigma=1)
    mz_img = NormalizeData(mz_img).ravel()
    pearson = pearsonr(x=img, y=mz_img)[0]
    cosine_sim = 1 - spatial.distance.cosine(img, mz_img)
    return pearson, cosine_sim
