import pyimzml.ImzMLParser as ImzMLParser
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import numpy as np
import os
import imzml_writer.utils as iw_utils
import time


def convert_from_RAW(dir:str,mode:str="Centroid",x_speed:float=40.0,y_step:float=150.0,filetype:str="raw"):
    """Placeholder"""
    iw_utils.RAW_to_mzML(dir,write_mode=mode)
    
    ##Waiting loop to check if msconvert has finished it's work:
    all_files = os.listdir(dir)
    num_raw_files = 0
    for file in all_files:
        if file.split(".")[-1] == filetype:
            num_raw_files+=1

    num_mzML = 0
    while num_mzML < num_raw_files:
        num_mzML = 0
        all_files = os.listdir(dir)
        for file in all_files:
            if file.split(".")[-1] == "mzML":
                num_mzML += 1
        time.sleep(1)

    time.sleep(5)

    iw_utils.clean_raw_files(dir,filetype)
    mzML_path = os.path.join(dir,"Output mzML Files")
    iw_utils.mzML_to_imzML_convert(PATH=mzML_path)

    iw_utils.imzML_metadata_process(
        model_files=mzML_path,
        x_speed=x_speed,
        y_step=y_step,
        path=dir
        )
    




def get_image_matrix(src:str, mz:list | float = 104.1070,tol: list | float = 10.0):
    """Placeholder for now"""

    with warnings.catch_warnings(action="ignore"):
        with ImzMLParser.ImzMLParser(filename=src,parse_lib='lxml') as img:
            if isinstance(mz,float):
                tolerance = mz * tol / 1e6
                img_raw = ImzMLParser.getionimage(img, mz, tolerance)
            elif isinstance(mz,list):
                img_raw = []
                for idx, spp in enumerate(mz):
                    if isinstance(tol,float):
                        tolerance = spp * tol / 1e6
                    elif isinstance(tol,list):
                        tolerance = spp * tol[idx] / 1e6
                    img_raw.append(ImzMLParser.getionimage(img,spp,tolerance))
                
    return img_raw


def get_TIC_image(src:str):
    """Placeholder"""
    with warnings.catch_warnings(action='ignore'):
        with ImzMLParser.ImzMLParser(filename=src,parse_lib='lxml') as img:
            tic_image = ImzMLParser.getionimage(img,500,9999)
    
    return tic_image
    

def get_scale(src:str):
    """Placeholder"""
    with warnings.catch_warnings(action="ignore"):
        img = ImzMLParser.ImzMLParser(filename=src,parse_lib='lxml')
        metadata = img.metadata.pretty()
        scan_settings = metadata["scan_settings"]["scanSettings1"]
        for key in scan_settings.keys():
            if key == "max dimension x":
                scale_x = scan_settings[key]
            elif key == "max dimension y":
                scale_y = scan_settings[key]
        return scale_x, scale_y

def get_aspect_ratio(src:str):
    """Placeholder"""
    with warnings.catch_warnings(action="ignore"):
        img = ImzMLParser.ImzMLParser(filename=src,parse_lib='lxml')
        metadata = img.metadata.pretty()
        scan_settings = metadata["scan_settings"]["scanSettings1"]
        for key in scan_settings.keys():
            if key == "pixel size (x)" or key == "pixel size x":
                x_pix = scan_settings[key]
            elif key == "pixel size y":
                y_pix = scan_settings[key]
        
        return y_pix / x_pix


def draw_ion_image(data:np.array, cmap:str="viridis",mode:str = "draw", path:str = None, cut_offs:tuple=(5, 95),quality:int=100, asp:float=1,scale:float=1):
    mpl.rcParams['savefig.pad_inches'] = 0
    up_cut = np.percentile(data,max(cut_offs))
    down_cut = np.percentile(data,min(cut_offs))

    img_cutoff = np.where(data > up_cut,up_cut,data)
    img_cutoff = np.where(data < down_cut,0,data)

    fig = plt.figure()
    _plt = plt.subplot()
    _plt.axis('off')
    _plt.imshow(img_cutoff,aspect=asp,interpolation="none",cmap=cmap,vmax=up_cut,vmin=0)
    size = fig.get_size_inches()
    scaled_size = size * scale
    fig.set_size_inches(scaled_size)
    if mode == "draw":
        plt.show()
    elif mode == "save":
        if path is None:
            raise Exception("No file name specified")
        else:
            
            fig.savefig(path, dpi=quality,pad_inches=0,bbox_inches='tight')
    





    
