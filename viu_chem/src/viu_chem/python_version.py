import os
import subprocess
import time
import jm_helpers
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter
import numpy as np
import tifffile
import pandas as pd
import pathlib

##Target directory mode
TARGET_DIR = "/Users/josephmonaghan/Documents/msi_flow_datatest/msi"
TARGET_DIR = "/Users/josephmonaghan/Downloads/nametest/TF34I_KNJM_Jan25-2025_/quickcheck"
src_sheet = '/Users/josephmonaghan/Documents/JM Images/Lum2Tumors/IMAGING CAMPAIGN/Seg_flow.xlsx'

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

def files_from_sheet(sheet:str):
    files = pd.read_excel(src_sheet,header=None)[0].to_list()

    filenames = []
    dirs= []
    for file in files:
        file = file.replace("'","")
        tmp = pathlib.Path(file)
        filenames.append(tmp.name)
        dirs.append(tmp.parent)
        
    
    return filenames, dirs



def main(mode:str):
    if mode == "sheet":
        imzML_files, directories = files_from_sheet("xyz")
    elif mode == "dir":
        imzML_files, directories = files_from_dir(TARGET_DIR)
    
    #Calls the peakpicking algorithm
    for idx, file in enumerate(imzML_files):
        tgt_path = os.path.join(directories[idx],file)
        print(tgt_path)
        subprocess.run(["python3", "msi_preprocessing_flow/scripts/peak_picking/peak_picking.py",tgt_path])
        
        work_dir = os.path.join(directories[idx],"peakpicking")
        subprocess.run(["python3", "msi_preprocessing_flow/scripts/alignment/get_reference_spectrum.py",work_dir])
        cmz_loc = os.path.join(work_dir,"alignment","cmz.npy")
        tgt_path = os.path.join(work_dir,file)
        subprocess.run(["python3", "msi_preprocessing_flow/scripts/alignment/alignment.py",tgt_path,cmz_loc])
        work_dir = os.path.join(work_dir,"alignment")
        tgt_path = os.path.join(work_dir,file)
        ##Does single sample segmentation
        subprocess.run(["python3","msi_segmentation_flow/scripts/single_sample_segmentation.py",tgt_path,'-matrix_cluster',"True","-n_neighbors", "100"])
        cur_files = os.listdir(work_dir)
        for loc_file in cur_files:
            if loc_file.startswith("umap"):
                img_dir = os.path.join(work_dir,loc_file,"binary_imgs")
                break

        ##Eventually should read that segmentation to pick out matrix vs. sample
        subprocess.run(["python3","msi_preprocessing_flow/scripts/matrix_removal/get_matrix_pixels_from_segmentation.py",tgt_path,img_dir])

        #Actually removes the pixels
        files = os.listdir(os.path.join(work_dir,"matrix_removal"))
        for loc_file in files:
            if loc_file.startswith(file.split(".")[0]):
                matrix_img = os.path.join(work_dir,"matrix_removal", loc_file)
        subprocess.run(["python3","msi_preprocessing_flow/scripts/matrix_removal/matrix_removal.py",tgt_path,matrix_img])

        path = os.path.join(directories[idx],file)
        parser = ImzMLParser(path)
        matrix_img_loc = os.path.join(work_dir,"matrix_removal",f"{file.split(".")[0]}_postproc_matrix_image.tif")
        img = tifffile.imread(matrix_img_loc)
        img = img - img.min()
        img = (img / img.max()).astype(int)
        img = img > 0

        num_pixels = len(img.flatten())
        bin_img_px_idx_np = np.nonzero(img)
        bin_img_px_idx = tuple(zip(bin_img_px_idx_np[1], bin_img_px_idx_np[0]))

        replace_file = os.path.join(directories[idx],"tmp.imzML")
        new_file = ImzMLWriter(replace_file)
        for i in range(num_pixels):
            coords = parser.coordinates[i]
            search_coords = (coords[0], coords[1])
            if search_coords not in bin_img_px_idx:
                spectrum = parser.getspectrum(i)
                new_file.addSpectrum(spectrum[0],spectrum[1],search_coords)
                
        new_file.close()

        src = os.path.join(directories[idx],file)
        # needs_work = os.path.join(work_dir,"matrix_removal",file)
        needs_work = os.path.join(directories[idx],"tmp.imzML")
        dest = needs_work
        jm_helpers.reannotate_imzML(needs_work,src,dest)





    # #Generate coherent mz list for overall expt
    # work_dir = os.path.join(TARGET_DIR,"peakpicking")
    # subprocess.run(["python3", "msi_preprocessing_flow/scripts/alignment/get_reference_spectrum.py",work_dir])
    # cmz_loc = os.path.join(work_dir,"alignment","cmz.npy")

    # for file in imzML_files:
    #     tgt_path = os.path.join(work_dir,file)
    #     subprocess.run(["python3", "msi_preprocessing_flow/scripts/alignment/alignment.py",tgt_path,cmz_loc])

    # work_dir = os.path.join(work_dir,"alignment")


    # for file in imzML_files:
    #     tgt_path = os.path.join(work_dir,file)
    #     ##Does single sample segmentation
    #     subprocess.run(["python3","msi_segmentation_flow/scripts/single_sample_segmentation.py",tgt_path,'-matrix_cluster',"True","-n_neighbors", "100"])
    #     cur_files = os.listdir(work_dir)
    #     for loc_file in cur_files:
    #         if loc_file.startswith("umap"):
    #             img_dir = os.path.join(work_dir,loc_file,"binary_imgs")
    #             break

    #     ##Eventually should read that segmentation to pick out matrix vs. sample
    #     subprocess.run(["python3","msi_preprocessing_flow/scripts/matrix_removal/get_matrix_pixels_from_segmentation.py",tgt_path,img_dir])

    #     #Actually removes the pixels
    #     files = os.listdir(os.path.join(work_dir,"matrix_removal"))
    #     for loc_file in files:
    #         if loc_file.startswith(file.split(".")[0]):
    #             matrix_img = os.path.join(work_dir,"matrix_removal", loc_file)
    #     subprocess.run(["python3","msi_preprocessing_flow/scripts/matrix_removal/matrix_removal.py",tgt_path,matrix_img])

    #     #Read old imzML to get pixel dimensions, grab matrix array from above, and check coords, writing pixel by pixel


    # for file in imzML_files:
    #     path = os.path.join(TARGET_DIR,file)
    #     parser = ImzMLParser(path)
    #     matrix_img_loc = os.path.join(work_dir,"matrix_removal",f"{file.split(".")[0]}_postproc_matrix_image.tif")
    #     img = tifffile.imread(matrix_img_loc)
    #     img = img - img.min()
    #     img = (img / img.max()).astype(int)
    #     img = img > 0

    #     num_pixels = len(img.flatten())
    #     bin_img_px_idx_np = np.nonzero(img)
    #     bin_img_px_idx = tuple(zip(bin_img_px_idx_np[1], bin_img_px_idx_np[0]))

    #     replace_file = os.path.join(TARGET_DIR,"tmp.imzML")
    #     new_file = ImzMLWriter(replace_file)
    #     for i in range(num_pixels):
    #         coords = parser.coordinates[i]
    #         search_coords = (coords[0], coords[1])
    #         if search_coords not in bin_img_px_idx:
    #             spectrum = parser.getspectrum(i)
    #             new_file.addSpectrum(spectrum[0],spectrum[1],search_coords)
        

    #     new_file.close()

    # for file in imzML_files:
        # src = os.path.join(TARGET_DIR,file)
        # # needs_work = os.path.join(work_dir,"matrix_removal",file)
        # needs_work = os.path.join(TARGET_DIR,"tmp.imzML")
        # dest = needs_work
        # jm_helpers.reannotate_imzML(needs_work,src,dest)


if __name__=="__main__":
    main(mode="sheet")








