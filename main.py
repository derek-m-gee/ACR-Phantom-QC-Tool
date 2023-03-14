# Prototyping for DICOM thresholding
#References:
#https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/
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

#Finds slices with max intensity >300 HU but no more than 800
# Returns slice with fewest number of non_zero
def find_slice(stack, min_val=300, max_val =800):
    possible_slice = []
    n_nonzero = []
    slice_max = []
    for s in np.arange(len(stack)):
        mask = img_threshold(stack[s])
        if np.count_nonzero(mask > min_val) > 0:
            if mask.max() < max_val:
                possible_slice.append(s)
                n_nonzero.append(np.count_nonzero(mask))
                slice_max.append(mask.max())
    probable_slice = possible_slice[n_nonzero.index(min(n_nonzero))]
    max_HU = slice_max[n_nonzero.index(min(n_nonzero))]
    if (probable_slice-1 in possible_slice):
        check_max = stack[probable_slice-1].max()
        if ((check_max-max_HU)/max_HU < .1):
            probable_slice -= 1
    if (probable_slice+1 in possible_slice):
        check_max = stack[probable_slice+1].max()
        if ((check_max-max_HU)/max_HU < .1):
            probable_slice += 1
    return probable_slice

def add_ROI(slice,xpos,ypos,pixel_size,area=200):
    mask = np.zeros(slice.shape, np.uint8)
    center = (int(cy_phantom - ypos / pixel_size[0]), int(cx_phantom + xpos / pixel_size[1]))
    radius = int(math.sqrt(area/math.pi) / pixel_size[0])
    cv.circle(mask, center, radius, 255, 3)
    indices = np.array(np.transpose(np.nonzero(mask)))
    return indices

def window_image(img, level,window):
    img_min = level - window // 2
    img_max = level + window // 2
    imageWL = img.copy()
    imageWL[imageWL < img_min] = img_min
    imageWL[imageWL > img_max] = img_max
    return imageWL

if __name__ == '__main__':
    id=0
    data_path = r"C:\Users\derek\Documents\RAMD5394IndependentStudy\Test_Project\ADULT_ABD_10\\"
    #data_path = r"C:\Users\derek\Documents\RAMD5394IndependentStudy\Test_Project\ADULT_ABD_12\\"
    output_path = working_path = r"C:\Users\derek\Documents\RAMD5394IndependentStudy\Test_Project\\"
    patient = load_scan(data_path)
    patient_HU = get_pixels_hu_stack(patient)
    pixel_size = patient[0].PixelSpacing
    #Get slice
    #patient_slice = patient_HU[0,:,:]

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
    module_3_index = find_slice(img_stack)
    module_3 = img_stack[module_3_index]
    module_3_pos = module_3_index * patient[0].SliceThickness #mm
    module_1_index = int((module_3_pos - 80)/patient[0].SliceThickness)
    module_1 = img_stack[module_1_index]
    module_2_index = int((module_3_pos - 40)/patient[0].SliceThickness)
    module_2 = img_stack[module_2_index]
    # plt.imshow(module_1, cmap='gray')
    # plt.show()

    #######################################################################################
    # Phantom Angle Correction
    #######################################################################################

    module_3BBs = img_threshold(module_3.copy(),.6)
    mod_3BB_8bit = cv.convertScaleAbs(module_3BBs,alpha=(255/32768))
    BB_contours, hierarchy = cv.findContours(mod_3BB_8bit, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    BBx = []
    BBy = []
    for i in np.arange(len(BB_contours)):
        M = cv.moments(BB_contours[i])
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        BBx.append(cx)
        BBy.append(cy)
    BB_centers = list(zip(BBx,BBy))
    p1 = BB_centers[0]
    p2 = BB_centers[1]
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    ang = np.rad2deg(np.arctan2(dy,dx))
    offset = 45 + ang

    height,width = mod_3BB_8bit.shape[:2]
    rotation_matrix = cv.getRotationMatrix2D((width/2, height/2), offset, 1)
    for i in np.arange(len(img_stack)):
        img_stack[i] = cv.warpAffine(img_stack[i], rotation_matrix, (width, height))

    #######################################################################################
    # Module 1: CT Number Accuracy Operations
    # Phantom Edge Detection and Centering
    #######################################################################################

    mod_1_8bit = patient[module_1_index].pixel_array.copy()
    #mod_1_8bit = cv.warpAffine(mod_1_8bit, rotation_matrix, (width, height))
    mod_1_8bit = cv.convertScaleAbs(mod_1_8bit,alpha=(255/32768))
    edges = cv.Canny(mod_1_8bit,1,6)
    # plt.imshow(edges, cmap='gray')
    # plt.show()

    #findContours automatically finds contours but finds all contours including objects inlaid or connected
    # the RETR_EXTERNAL contouring mode only finds the outer or parent objects
    # this only works at the moment due to the Canny filter not detecting the complete phantom edge
    #contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    kernel = np.ones((5, 5), np.uint8)
    mod_1_erosion = cv.erode(mod_1_8bit,kernel,iterations=1)
    mod_1_dilation = cv.dilate(mod_1_erosion,kernel,iterations=1)
    ret,thresh1 = cv.threshold(mod_1_dilation,4,255,cv.THRESH_BINARY)
    kernel = np.ones((3,3),np.uint8)
    img_erosion = cv.erode(thresh1,kernel,iterations=1)
    phantom_edges = thresh1 - img_erosion
    # plt.imshow(phantom_edges,cmap='gray')
    # plt.show()
    contours, hierarchy = cv.findContours(phantom_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour_size = [len(cntrs) for cntrs in contours]
    phantom_contour = contours[contour_size.index(max(contour_size))]
   #Finds centroid for only the largest contour, the phantom's edge
    M = cv.moments(phantom_contour)
    cx_phantom = int(M['m10'] / M['m00'])
    cy_phantom = int(M['m01'] / M['m00'])

    contour_list = []
    insert_ctrs = []
    contour_list.append(add_ROI(module_1,45,44,pixel_size))
    insert_ctrs.append(len(contour_list) - 1)
    contour_list.append(add_ROI(module_1,45, -44,pixel_size))
    insert_ctrs.append(len(contour_list) - 1)
    contour_list.append(add_ROI(module_1,-45, 44,pixel_size))
    insert_ctrs.append(len(contour_list) - 1)
    contour_list.append(add_ROI(module_1,-45, -44,pixel_size))
    insert_ctrs.append(len(contour_list) - 1)
    contour_list.append(add_ROI(module_1,-60, 0, pixel_size))
    insert_ctrs.append(len(contour_list) - 1)

    # contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # #sorts contours by determining if they are an insert
    # # contours fused to determine centroid of phantom
    # contour_list = list(contours)
    # insert_ctrs = []
    # fused_contour_pts = []
    # for i in np.arange(len(contours)):
    #     if (contours[i].size > 100) and (contours[i].size < 250):
    #         insert_ctrs.append(i)
    #     elif contours[i].size >250:
    #         fused_contour_pts += [pt[0] for pt in contours[i]]
    # fused_contour = np.array(fused_contour_pts).reshape((-1,1,2)).astype(np.int32)
    #
    # pixel_size = ds.PixelSpacing
    # while len(insert_ctrs) < 4:
    #     for i in np.arange(len(insert_ctrs)):
    #         ROIx = []
    #         ROIy = []
    #         flat_list = np.ndarray.flatten(contours[insert_ctrs[i]])
    #         #iterates over every x-coordinate in the flattened list of ordered pairs
    #         for x in range(0,len(flat_list),2):
    #             #If x-coordinate is left of the center, copy right
    #             #If x-coordinate is right of the center, copy left
    #             if flat_list[x]<cx_phantom:
    #                 ROIx.append(int(flat_list[x]+87/pixel_size[1]))
    #                 ROIy.append(flat_list[x+1])
    #             elif flat_list[x]>cx_phantom:
    #                 ROIx.append(int(flat_list[x]-87/pixel_size[1]))
    #                 ROIy.append(flat_list[x + 1])
    #             # elif flat_list[x+1]<cy_phantom:
    #             #     ROIy.append(int(flat_list[x+1]+87/pixel_size[0]))
    #             # else:
    #             #     ROIy.append(int(flat_list[x+1]-87/pixel_size[0]))
    #         new_ROI = list(zip(ROIx,ROIy))
    #         contour_list.append(np.array(new_ROI))
    #         insert_ctrs.append(len(contour_list)-1)
    #         if len(insert_ctrs) == 4:
    #             mask = np.zeros(module_1.shape,np.uint8)
    #             center = (int(height/2),int(width/2-50/pixel_size[1]))
    #             radius = int(80*pixel_size[0])
    #             cv.circle(mask,center,radius,255,3)
    #             indices = np.transpose(np.nonzero(mask))
    #             contour_list.append(np.array(indices))
    #             insert_ctrs.append(len(contour_list) - 1)
#-100/pixel_size[1]
    #Draws the insert ROIs
    # drawing_img = np.zeros_like(edges)
    # for i in insert_ctrs:
    #     cv.drawContours(drawing_img, contour_list, i, 255, 3)
    # plt.imshow(drawing_img, cmap='gray')
    # plt.show()

    #Draw Original ROIs on phantom
    # for i in insert_ctrs:
    #     cv.drawContours(module_1, contour_list, i, 255, 3)
    # plt.imshow(module_1, cmap='gray')
    # plt.show()

    #should make an insert class and instantiate objects
    bone = []
    air = []
    acrylic = []
    polyethylene = []
    water = []
    unknown = []
    module_1_Dislplay = window_image(module_1.copy(),0,400)
    for i in insert_ctrs:
        mask = np.zeros(module_1.shape,np.uint8)
        cv.drawContours(mask,contour_list,i,255,-1)
        # area = pixel_size[0]*pixel_size[1]*np.count_nonzero(mask)
        # #Iteratively reduce ROI area to match ACR guidelines for ~200 mm2 ROI
        # #These bounds allow for up to +/-2.5% ROI area
        # while area > 210 or area < 190:
        #     if area < 190:
        #         kernel = np.ones((3, 3), np.uint8)
        #         mask = cv.dilate(mask, kernel, iterations=1)
        #         area = pixel_size[0]*pixel_size[1]*np.count_nonzero(mask)
        #     else:
        #         kernel = np.ones((3, 3), np.uint8)
        #         mask = cv.erode(mask, kernel, iterations=5)
        #         area = pixel_size[0]*pixel_size[1]*np.count_nonzero(mask)

        ROI_val = cv.mean(module_1,mask)[0]
        if ROI_val >= 750: #bone
            bone.append(ROI_val)
            if ROI_val > 850 and ROI_val < 970:
                bone.append(1)
            else:
                bone.append(0)
        elif ROI_val <= -750: #air
            air.append(ROI_val)
            if ROI_val > -1005 and ROI_val < -970:
                air.append(1)
            else:
                air.append(0)
        elif ROI_val > 50 and ROI_val < 300: #acrylic
            acrylic.append(ROI_val)
            if ROI_val > 110 and ROI_val < 135:
                acrylic.append(1)
            else:
                acrylic.append(0)
        elif ROI_val > -300 and ROI_val < -50: #polyethylene
            polyethylene.append(ROI_val)
            if ROI_val > -107 and ROI_val < -84:
                polyethylene.append(1)
            else:
                polyethylene.append(0)
        elif ROI_val > -50 and ROI_val < 50: #water
            water.append(ROI_val)
            if ROI_val > -7 and ROI_val < 7:
                water.append(1)
            else:
                water.append(0)
        else: #error, check ROI positioning/scan protocol
            print("ROI Tool failed to detect material")
            print(ROI_val)
            unknown.append(ROI_val)

        contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(module_1_Dislplay, contours, -1, 255, 3)
    plt.imshow(module_1_Dislplay, cmap='gray')
    plt.show()

    contour_list = []
    insert_ctrs = []
    contour_list.append(add_ROI(module_2,0,58,pixel_size,100))
    insert_ctrs.append(len(contour_list) - 1)
    contour_list.append(add_ROI(module_2,-20, 58,pixel_size,100))
    insert_ctrs.append(len(contour_list) - 1)

    module_2_index = int((module_3_pos - 40)/patient[0].SliceThickness)
    scan_type = 'adult'
    max_CNR = 0
    j = 1
    plt.figure()
    for test_slice in np.arange(module_2_index-5,module_2_index+7,1):
        module_2 = window_image(img_stack[test_slice],100,100)
        ROI_val = []
        module_2_Dislplay = module_2.copy()
        for i in insert_ctrs:
            mask = np.zeros(module_2.shape,np.uint8)
            cv.drawContours(mask,contour_list,i,255,-1)
            mean, std_dev = cv.meanStdDev(module_2,mask=mask)
            ROI_val.append([mean.item(),std_dev.item()])
            contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(module_2_Dislplay, contours, -1, 255, 3)
        plt.subplot(2,6, j)
        plt.imshow(module_2_Dislplay, cmap='gray')
        plt.title('Slice: {}'.format(test_slice))
        plt.axis('off')
        j=j+1
        CNR = abs(ROI_val[0][0]-ROI_val[1][0])/ROI_val[1][1]
        if CNR > max_CNR:
            max_CNR = CNR
            module_2_index = test_slice
            module_2_CNR_disp = module_2_Dislplay.copy()
    plt.show()

    if max_CNR > 1 and scan_type == 'adult':
        test_pass = 'PASS'
    elif max_CNR > 0.4 and scan_type == "ped_abd":
        test_pass = 'PASS'
    elif max_CNR > 0.7 and scan_type == 'ped_head':
        test_pass = 'PASS'
    else:
        test_pass = 'FAIL'

    plt.figure()
    plt.imshow(module_2_CNR_disp, cmap='gray')
    plt.title('Slice: {}  CNR:{}  {}'.format(module_2_index,round(max_CNR,4),test_pass))
    plt.axis('off')
    plt.show()

    contour_list = []
    insert_ctrs = []
    contour_list.append(add_ROI(module_3,0,0,pixel_size,400))
    insert_ctrs.append(len(contour_list) - 1)
    contour_list.append(add_ROI(module_3,0,68,pixel_size,400))
    insert_ctrs.append(len(contour_list) - 1)
    contour_list.append(add_ROI(module_3,0, -68,pixel_size,400))
    insert_ctrs.append(len(contour_list) - 1)
    contour_list.append(add_ROI(module_3,68,0,pixel_size,400))
    insert_ctrs.append(len(contour_list) - 1)
    contour_list.append(add_ROI(module_3,-68,0,pixel_size,400))
    insert_ctrs.append(len(contour_list) - 1)

    ROI_val = []
    module_3_Dislplay = module_3.copy()
    module_3_Dislplay = window_image(module_3_Dislplay,0,100)
    for i in insert_ctrs:
        mask = np.zeros(module_3.shape, np.uint8)
        cv.drawContours(mask, contour_list, i, 255, -1)
        mean, std_dev = cv.meanStdDev(module_3, mask=mask)
        ROI_val.append([mean.item(), std_dev.item()])
        contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(module_3_Dislplay, contours, -1, 255, 3)
    center = ROI_val[0]
    diff = []
    test_uniformity = []
    for pos in np.arange(1,len(ROI_val)):
        diff.append(ROI_val[pos][0] - ROI_val[0][0])
        if diff[len(diff)-1] > 7 or diff[len(diff)-1] < -7:
            test_uniformity.append('FAIL')
        elif diff[len(diff)-1] > 5 or diff[len(diff)-1] < -5:
            test_uniformity.append('MINOR DEFICIENCY')
        else:
            test_uniformity.append('PASS')

    plt.imshow(module_3_Dislplay, cmap='gray')
    plt.title('12: {}  3: {}  6: {}  9: {}'
              '\n Center:{}'
              '\nDifference from Center {} {} {} {}'
              '\nUniformity within +/- 7 {} {} {} {}'.format(round(ROI_val[1][0],3),round(ROI_val[3][0],3),round(ROI_val[2][0],3),
                                                  round(ROI_val[4][0],3),round(ROI_val[0][0],3),round(diff[0],2),round(diff[2],2),round(diff[1],2),round(diff[3],2),
                                                                     test_uniformity[0], test_uniformity[2],test_uniformity[1],test_uniformity[3]))
    plt.axis('off')
    plt.show()

    pass