# Automated tool for ACR CT Image Quality QC
#
#References:
#https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/
#https://www.researchgate.net/publication/9063443_Accuracy_of_a_simple_method_for_deriving_the_presampled_modulation_transfer_function_of_a_digital_radiographic_system_from_an_edge_image
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2 as cv
from imageTools import *

if __name__ == '__main__':
    id=0
    # data_path = r"C:\Users\derek\Documents\RAMD5394IndependentStudy\ACRQCtool\ADULT_ABD_10\\"
    data_path = r"C:\Users\derek\Documents\RAMD5394IndependentStudy\ACRQCtool\ADULT_ABD_12\\"
    # data_path = r"C:\Users\derek\Documents\RAMD5394IndependentStudy\ACRQCtool\80_kVp_3_mm_4\\"
    # data_path = r"C:\Users\derek\Documents\RAMD5394IndependentStudy\ACRQCtool\PED_BODY_6\\"

    output_path = working_path = r"C:\Users\derek\Documents\RAMD5394IndependentStudy\Test_Project\\"
    patient = load_scan(data_path)
    patient_HU = get_pixels_hu_stack(patient)
    pixel_size = patient[0].PixelSpacing

    np.save(output_path + "fullimages_{}.npy".format(id), patient_HU)
    file = output_path + "fullimages_{}.npy".format(id)
    img_stack = np.load(file).astype(np.float64)
    #histogram_from_stack(img_stack)
    #display_stack(img_stack)
    #display_stack(img_threshold(img_stack))

    #######################################################################################
    # Module 1 Indexing
    # Uses Module 3 BBs as landmark to determine Module 1 slice position.
    # ACR Phantom Modules 40 mm in depth and 200 mm diameter.
    # subtracting the length of 2 modules from the position of the
    # known module 3 slice position gives the position of module 1.
    #######################################################################################
    from module3 import find_slice
    module_3_index = find_slice(img_stack)
    module_3 = img_stack[module_3_index]
    module_3_pos = module_3_index * patient[0].SliceThickness #mm
    module_1_index = int((module_3_pos - 80)/patient[0].SliceThickness)
    module_1 = img_stack[module_1_index]
    module_2_index = int((module_3_pos - 40)/patient[0].SliceThickness)
    module_2 = img_stack[module_2_index]
    module_4_index = int((module_3_pos + 40)/patient[0].SliceThickness)
    module_4 = img_stack[module_4_index]
    # plt.imshow(module_1, cmap='gray')
    # plt.show()

    #######################################################################################
    # Phantom Angle Correction
    #######################################################################################
    from module3 import find_rotation, get_BBs
    BB_contours = get_BBs(module_3)
    offset = find_rotation(BB_contours,45)

    height,width = module_3.shape[:2]
    rotation_matrix = cv.getRotationMatrix2D((width/2, height/2), offset, 1)
    for i in np.arange(len(img_stack)):
        img_stack[i] = cv.warpAffine(img_stack[i], rotation_matrix, (width, height))

    #######################################################################################
    # Module 1: CT Number Accuracy Operations
    # Phantom Edge Detection and Centering
    #######################################################################################
    from module1 import *
    #needs to be rotated again since this data is from the patient set, not the img_stack
    mod_1_8bit = patient[module_1_index].pixel_array.copy()
    mod_1_8bit = cv.warpAffine(mod_1_8bit, rotation_matrix, (width, height))
    mod_1_8bit = cv.convertScaleAbs(mod_1_8bit,alpha=(255/32768))
    # plt.imshow(edges, cmap='gray')
    # plt.show()
    cx_phantom, cy_phantom = phantom_center(mod_1_8bit)

    #Allows for adjusting of ROI placement
    dx = 44.25
    dy = 43.75
    #Requires module 1 in HU
    bone, air, acrylic, polyethylene, water, unknown = sample_inserts(module_1,cx_phantom,dx,cy_phantom,dy,pixel_size)

    #######################################################################################
    # Module 2: CNR
    #######################################################################################
    from module2 import get_CNR
    CNR = get_CNR(img_stack,module_2,module_2_index,cx_phantom,cy_phantom,pixel_size,'adult')

    #######################################################################################
    # Module 3: Uniformity
    #######################################################################################
    from module3 import test_uniformity
    test_uniformity(module_3, cx_phantom, cy_phantom, pixel_size)

    #######################################################################################
    # Module 4: Spatial Resolution
    #######################################################################################
    from module4 import *
    max_res = test_res(module_4,cx_phantom,cy_phantom,pixel_size)

    pass