# Automated tool for ACR CT Image Quality QC
#
#References:
#https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/
#https://www.researchgate.net/publication/9063443_Accuracy_of_a_simple_method_for_deriving_the_presampled_modulation_transfer_function_of_a_digital_radiographic_system_from_an_edge_image
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import pydicom
import cv2 as cv

#Loads in all DICOM
def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] -
                                 slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices

#Pixel array stores data as intensity, this function linearly converts
# to Hounsfield Units using slope and intercept from DICOM Header
def get_pixels_hu(ds):
    image_data = ds.pixel_array
    image_data = image_data.astype(np.int16)
    # Set outside-of-scan pixels to 1
    # Intercept is usually -1024, so air is approximately 0
    image_data[image_data == -2000] = 0

    # Convert to Hounsfield Units (HU)
    intercept = ds.RescaleIntercept
    slope = ds.RescaleSlope

    if slope != 1:
        image_data = slope * image_data.astype(np.float64)
        image_data = image_data.astype(np.int16)
    image_data += np.int16(intercept)
    return np.array(image_data, dtype=np.int16)

def get_pixels_hu_stack(scans):
    image_data = np.stack([s.pixel_array for s in scans])
    image_data = image_data.astype(np.int16)
    # Set outside-of-scan pixels to 1
    # Intercept is usually -1024, so air is approximately 0
    image_data[image_data == -2000] = 0

    # Convert to Hounsfield Units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image_data = slope * image_data.astype(np.float64)
        image_data = image_data.astype(np.int16)
    image_data += np.int16(intercept)
    return np.array(image_data, dtype=np.int16)

#Displays histogram data
def histogram_from_stack(img, id=0):
    plt.hist(img.flatten(),bins=50,color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()
    return

def display_stack(stack,start_with=0,show_every=3):
    rows = math.isqrt(len(stack))
    cols = math.isqrt(len(stack))
    fig,ax = plt.subplots(int(rows/show_every),cols,figsize=[12,12])
    for i in range(rows*cols):
        try:
            ind = start_with + i*show_every
            ax[int(i/rows),int(i % rows)].set_title('slice {}'.format(ind))
            ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')
            ax[int(i / rows), int(i % rows)].axis('off')
        except:
            pass
    plt.show()

#Thresholds image/stack for high intensity objects, level sets %limit for threshold
def img_threshold(img,level = 0.8):
    #Arbitrary Global Thresholding
    ret, th = cv.threshold(img,img.max()*(1-level),img.max(),cv.THRESH_BINARY)

    #Adaptive Thresholding
    #ret, th = cv.adaptiveThreshold(img,1500,cv.ADAPTIVE_THRESH_MEAN_C,
    #        cv.THRESH_BINARY,11,2)
    #Otsu Binarization
    #ret, th = cv.threshold(img, 0, 1500, cv.THRESH_BINARY+cv.THRESH_OTSU)

    #plt.imshow(th, cmap='gray')
    #plt.show()
    return th

def window_image(img, level,window):
    img_min = level - window // 2
    img_max = level + window // 2
    imageWL = img.copy()
    imageWL[imageWL < img_min] = img_min
    imageWL[imageWL > img_max] = img_max
    return imageWL
def add_ROI(slice,cx,xpos,cy, ypos,pixel_size,area=200):
    mask = np.zeros(slice.shape, np.uint8)
    center = (int(cy - ypos / pixel_size[0]), int(cx + xpos / pixel_size[1]))
    #260 255
    radius = int(math.sqrt(area/math.pi) / pixel_size[0])
    cv.circle(mask, center, radius, 255, 3)
    indices = np.array(np.transpose(np.nonzero(mask)))
    return indices
#Samples the image across line specified by start and end points
def get_LP(img,coord,width=1):
    line_mask=np.zeros_like(img)
    cv.line(line_mask,coord[0],coord[1],255,thickness=width)
    line_profile = cv.bitwise_and(img,line_mask)
    return line_profile
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
    #######################################################################################

    #ACR Phantom Modules 40 mm in depth and 200 mm diamter
    # subtracting the length of 2 modules from the position of the
    # known module 3 slice position gives the position of module 1.
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
    from module3 import find_rotation
    module_3BBs = img_threshold(module_3.copy(),.6)
    mod_3BB_8bit = cv.convertScaleAbs(module_3BBs,alpha=(255/32768))
    BB_contours, hierarchy = cv.findContours(mod_3BB_8bit, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    offset = find_rotation(BB_contours,45)

    height,width = mod_3BB_8bit.shape[:2]
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