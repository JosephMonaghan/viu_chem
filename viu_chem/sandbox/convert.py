from viu_chem import MSI_Process
import os
import viu_chem.python_version as msi_flow
import logging
import datetime
import shutil


start_time = datetime.datetime.now()

logger = logging.getLogger(__name__)

directory = '/Users/josephmonaghan/Documents/JM Images/Lum2Tumors/IMAGING CAMPAIGN/LUM2_BulkProcess'

all_folders = os.listdir(directory)
data_folders = []
for folder in all_folders:
    if not folder.startswith("."):
        data_folders.append(folder)


for idx, folder in enumerate(data_folders):
    full_path = os.path.join(directory,folder)

    now = datetime.datetime.now()
    logger.info(f"{now}: Starting conversion on file {idx+1} of {len(data_folders)}")
    logger.info(f"{now}: Folder is... {folder}")

    MSI_Process.convert_from_RAW(full_path)

now = datetime.datetime.now()
logger.info(f"{now}: All imzML writing complete in {now - start_time}, starting on msi_flow for matrix removal...")

bad_options = ["Initial RAW files", "Output mzML Files"]
end_string = "FTMS + p ESI Full ms [70.imzML"
for idx, folder in enumerate(data_folders):
    now = datetime.datetime.now()
    logger.info(f"{now}: Attempting msi_flow on folder ({idx+1} of {len(data_folders)}): {folder}")
    local_files = os.listdir(os.path.join(directory,folder))
    for dir in local_files:
        if not dir.startswith(".") and dir not in bad_options:
            msi_flow_folder = os.path.join(directory,folder,dir)
            for file in os.listdir(msi_flow_folder):
                if end_string in file:
                    ibd_file = file.split(".imzML")[0]+".ibd"
                    working_folder = os.path.join(msi_flow_folder,"mat_removal")
                    os.mkdir(working_folder)
                    shutil.copy(os.path.join(msi_flow_folder,file),working_folder)
                    shutil.copy(os.path.join(msi_flow_folder,ibd_file),working_folder)

                    msi_flow.preprocess(working_folder)
    
now = datetime.datetime.now()
logger.info(f"Finished! completed all in {now - start_time}")






