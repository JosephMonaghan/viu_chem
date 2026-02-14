import warnings
from pyimzml import ImzMLParser
from pathlib import Path

def get_mean_spectrum(img:Path,):
    """Extract mean spectrum from aligned imzML"""

    with warnings.catch_warnings(action='ignore'):
        imzml = ImzMLParser.ImzMLParser(img)
    
    metadata = imzml.metadata.pretty()
    is_continuous = metadata['file_description']['continuous']
    if not is_continuous:
        raise TypeError("imzML file must be continuous (aligned m/z)")
    
    mz, intensity = imzml.getspectrum(0)
    for idx, coord in enumerate(imzml.coordinates):
        if idx == 0:
            continue
        _, local_int = imzml.getspectrum(idx)
        intensity = local_int + intensity
    
    average_int = intensity / len(imzml.coordinates)
    return mz, average_int
    

