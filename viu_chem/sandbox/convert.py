from viu_chem import MSI_Process
import os

directory = "/Users/josephmonaghan/Downloads/Job_Queue"

all_folders = os.listdir(directory)
data_folders = []
for folder in all_folders:
    if not folder.startswith("."):
        data_folders.append(folder)


for idx, folder in enumerate(data_folders):
    full_path = os.path.join(directory,folder)

    print(f"Starting conversion on file {idx} of {len(data_folders)}")
    print(folder)
    MSI_Process.convert_from_RAW(full_path)




# MSI_Process.convert_from_RAW(directory)