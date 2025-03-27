# import viu_chem.python_version as msi_flow
# import os
# import imzml_writer.imzML_Scout as scout
import time
import datetime

start = datetime.datetime.now()
time.sleep(5)


end = datetime.datetime.now()
print(end - start)


# path = "/Users/josephmonaghan/Downloads/QuickCheck"
# # msi_flow.preprocess(path)

# #slow down to avoid error (test)
# time.sleep(1)

# files = os.listdir(path)
# for file in files:
#     if "_MatrixRemoved.imzML" in file:
#         path_img = os.path.join(path,file)
#         scout.main(path_img)
