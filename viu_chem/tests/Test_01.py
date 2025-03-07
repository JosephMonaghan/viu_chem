from viu_chem import MSI_Process as msi
import numpy as np

DATAFILE = "tests/Datafiles/Demo__FTMS + p ESI Full ms [70.imzML"
CHECKSUM = 19665189341.802734
MULTI_CHECKSUM = 20731235074.742188
ASPECT = 4.645352069116426

def test_matrix():
    assert np.sum(msi.get_image_matrix(DATAFILE)) == CHECKSUM

def multi_img():
    img_matrix = msi.get_image_matrix(src=DATAFILE,mz=[104.1070, 137.0709])
    multi_sum = 0
    for img in img_matrix:
        multi_sum += np.sum(img)
    
    assert multi_sum == MULTI_CHECKSUM
        
def test_img_draw():
    img = msi.get_image_matrix(DATAFILE)
    aspect_ratio = msi.get_aspect_ratio(DATAFILE)
    msi.draw_ion_image(img,"magma",mode='save',path="tests/images/test.tif",asp=aspect_ratio)

def check_aspect():
    assert msi.get_aspect_ratio(DATAFILE) == ASPECT
