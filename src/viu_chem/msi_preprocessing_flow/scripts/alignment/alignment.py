from pyimzml.ImzMLParser import ImzMLParser
from tqdm import tqdm
import numpy as np
import argparse
import os
from pyimzml.ImzMLWriter import ImzMLWriter


### pybasis function from https://bitbucket.org/iAnalytica/basis_pyproc/src/master/basis/preproc/palign.py
def pmatch_nn(refmz, mz, maxshift):

    """
    Performs nearest neighbour matching of mz or css feature vector to the reference one.

    Args:

        refmz: reference mz feature vector.

        mz: feature vector for alignment.

        maxshift: maximum allowed positional shift.

    Returns:

        refmzidcs: matched indices from refmz feature vector

        mzindcs: matached indices from mz feature vector

    """
    refmz = refmz.flatten()
    mz = mz.flatten()
    nvrbls = len(refmz)

    # map each mz to ref mz via interpolation
    mzindcs = np.round(np.interp(mz, refmz, np.arange(0., nvrbls)))
    mzindcs = (mzindcs.astype(int)).flatten()   # array with index of refmz for each mz

    # indices of mz values which are within maxshift tolerance to mapped refmz
    filtindcs = np.asarray(np.nonzero(np.abs(refmz[mzindcs] - mz) <= maxshift))
    filtindcs = (filtindcs.astype(int)).flatten()   # indices of refmz which have mappings within tolerance

    # count how many peaks are mapped to one ref m/z
    refmzidcs = np.unique(mzindcs[filtindcs])
    refmzidcs = np.asarray(refmzidcs)
    
    if len(refmzidcs)==0:
        return 0, 0
    mzbins = np.hstack([np.min(refmzidcs) - 0.5, refmzidcs.flatten() + .5])
    freq = np.histogram(mzindcs[filtindcs], bins=mzbins)
    mzrepidcs = refmzidcs[freq[0].astype(int) > 1]  # indices of refmz which has multiple mappings

    # for multiple mz mapping select mz which is closest to cmz
    mzfilt = mzindcs[filtindcs]     # indices of mz which were mapped within tolerance
    mz = mz[filtindcs]              # actual mz values which were mapped within tolerance
    uniqmzidx = (np.ones([1, len(mzfilt)])).flatten()
    for i in mzrepidcs:
        imzdx = ((np.asarray(np.nonzero(i == mzfilt))).astype(int)).flatten()
        minidx = (np.abs(mz[imzdx] - refmz[i])).argmin()
        uniqmzidx[imzdx] = 0.           # set all which are mapped to i to 0
        uniqmzidx[imzdx[minidx]] = 1.   # only set one which is closest to ref to 1

    uniqmzidx = (np.asarray(np.nonzero(uniqmzidx == 1.))).flatten()
    mzindcs = filtindcs[uniqmzidx]

    return refmzidcs, mzindcs


def align(imzML_fl:str,refmz:str,result_dir:str='',max_shift:float=0.05,debug:bool=False):
    if result_dir == '':
        result_dir = os.path.join(os.path.dirname(imzML_fl), "alignment")
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

    all_mzs = []
    all_ints = []

    # get common m/z vector
    cmz = np.load(refmz).astype(np.float32)

    # align all data to common m/z vector
    p = ImzMLParser(imzML_fl)
    with ImzMLWriter(os.path.join(result_dir, os.path.basename(imzML_fl))) as writer:
        for idx, (x, y, z) in enumerate(tqdm(p.coordinates)):
            mzs, intensities = p.getspectrum(idx)
            mzs = mzs.astype(np.float32)
            #cmz_intensities = get_ints_for_cmz(cmz, mzs, intensities)
            cmz_idx, matchmz_idx = pmatch_nn(cmz, mzs, max_shift)
            cmz_intensities = np.zeros(cmz.shape)
            cmz_intensities[cmz_idx] = intensities[matchmz_idx]
            writer.addSpectrum(cmz, cmz_intensities, (x, y, z))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Performs alignment to reference based on a nearest neighbor approach')
    parser.add_argument('imzML_fl', type=str, help='imzMl file')
    parser.add_argument('refmz', type=str, help='reference file as numpy array')
    parser.add_argument('-result_dir', type=str, default='', help='directory to store result, default \'\' to save results to directory called alignment')
    parser.add_argument('-max_shift', type=float, default=0.05, help='max mass shift in Da, default=0.05')
    parser.add_argument('-debug', type=bool, default=False, help='set to True for debugging')
    args = parser.parse_args()

    if args.result_dir == '':
        args.result_dir = os.path.join(os.path.dirname(args.imzML_fl), "alignment")
        if not os.path.exists(args.result_dir):
            os.mkdir(args.result_dir)

    all_mzs = []
    all_ints = []

    # get common m/z vector
    cmz = np.load(args.refmz).astype(np.float32)

    # align all data to common m/z vector
    p = ImzMLParser(args.imzML_fl)
    with ImzMLWriter(os.path.join(args.result_dir, os.path.basename(args.imzML_fl))) as writer:
        for idx, (x, y, z) in enumerate(tqdm(p.coordinates)):
            mzs, intensities = p.getspectrum(idx)
            mzs = mzs.astype(np.float32)
            #cmz_intensities = get_ints_for_cmz(cmz, mzs, intensities)
            cmz_idx, matchmz_idx = pmatch_nn(cmz, mzs, args.max_shift)
            cmz_intensities = np.zeros(cmz.shape)
            cmz_intensities[cmz_idx] = intensities[matchmz_idx]
            writer.addSpectrum(cmz, cmz_intensities, (x, y, z))

