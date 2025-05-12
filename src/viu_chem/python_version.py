import os
import viu_chem.jm_helpers as jm_helpers
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter
import numpy as np
import pandas as pd
import viu_chem.msi_preprocessing_flow as msi_preprocess
import viu_chem.msi_segmentation_flow as msi_segment
import tifffile


##Target directory mode

def files_from_dir(path:str):
    all_files = os.listdir(path)

    #Finds the imzML files
    imzML_files = []
    dirs = []
    for file in all_files:
        if file.split(".")[-1]=="imzML":
            imzML_files.append(file)
            dirs.append(path)
    
    return imzML_files, dirs

def preprocess(target_dir:str):
    imzML_files, directories = files_from_dir(target_dir)
    
    #Calls the peakpicking algorithm
    for idx, file in enumerate(imzML_files):
        tgt_path = os.path.join(directories[idx],file)
        print(tgt_path)
        work_dir = os.path.join(directories[idx],"peakpicking")

        os.makedirs(work_dir,exist_ok=True)
        msi_preprocess.peak_picking(tgt_path,work_dir)

        msi_preprocess.get_reference_spectrum(work_dir)
        cmz_loc = os.path.join(work_dir,"alignment","cmz.npy")
        tgt_path = os.path.join(work_dir,file)

        msi_preprocess.align(tgt_path,cmz_loc)

        work_dir = os.path.join(work_dir,"alignment")
        tgt_path = os.path.join(work_dir,file)
        # ##Does single sample segmentation

        msi_segment.segment(tgt_path,matrix_cluster=True,n_neighbors=100)
        cur_files = os.listdir(work_dir)
        for loc_file in cur_files:
            if loc_file.startswith("umap"):
                img_dir = os.path.join(work_dir,loc_file,"binary_imgs")
                break

        # ##Eventually should read that segmentation to pick out matrix vs. sample
        msi_preprocess.extract_matrix(tgt_path,img_dir)

        # #Actually removes the pixels
        files = os.listdir(os.path.join(work_dir,"matrix_removal"))
        for loc_file in files:
            if loc_file.startswith(file.split(".")[0]):
                matrix_img = os.path.join(work_dir,"matrix_removal", loc_file)

        msi_preprocess.matrix_removal(tgt_path,matrix_img)
        
        
        path = os.path.join(directories[idx],file)
        parser = ImzMLParser(path)
        
        matrix_img_loc = os.path.join(work_dir,"matrix_removal",f"{file.split(".")[0]}_postproc_matrix_image.tif")
        img = tifffile.imread(matrix_img_loc)
        img = np.delete(img,0,0)
        img = np.delete(img,0,1)
        img = img - img.min()
        img = (img / img.max()).astype(int)
        img = img > 0
        
        num_pixels = len(img.flatten())
        bin_img_px_idx_np = np.nonzero(img)
        bin_img_px_idx = tuple(zip(bin_img_px_idx_np[1], bin_img_px_idx_np[0]))

        new_file_name = file.split(".imzML")[0]+"_MatrixRemoved.imzML"
        replace_file = os.path.join(directories[idx],new_file_name)
        new_file = ImzMLWriter(replace_file,mode='processed')
        for i in range(num_pixels):
            coords = parser.coordinates[i]
            search_coords = (coords[0]-1, coords[1]-1)
            if search_coords not in bin_img_px_idx:
                search_coords = (coords[0], coords[1], 1)
                spectrum = parser.getspectrum(i)
                new_file.addSpectrum(spectrum[0],spectrum[1],search_coords)
                
        new_file.close()

        src = os.path.join(directories[idx],file)
        needs_work = replace_file
        dest = needs_work
        jm_helpers.reannotate_imzML(needs_work,src,dest)



