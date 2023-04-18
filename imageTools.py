import pydicom
import os
import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

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