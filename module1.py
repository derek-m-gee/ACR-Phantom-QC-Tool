# Module 1 Functions and Methods
import numpy as np
import math
import cv2 as cv
import matplotlib.pyplot as plt
from main import window_image, add_ROI

#findContours automatically finds contours but finds all contours including objects inlaid or connected
# the RETR_EXTERNAL contouring mode only finds the outer or parent objects
def phantom_center(mod_1):
    kernel = np.ones((5, 5), np.uint8)
    mod_1_erosion = cv.erode(mod_1, kernel, iterations=1)
    mod_1_dilation = cv.dilate(mod_1_erosion, kernel, iterations=1)
    ret, thresh1 = cv.threshold(mod_1_dilation, 4, 255, cv.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    img_erosion = cv.erode(thresh1, kernel, iterations=1)
    phantom_edges = thresh1 - img_erosion

    contours, hierarchy = cv.findContours(phantom_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contour_size = [len(cntrs) for cntrs in contours]
    phantom_contour = contours[contour_size.index(max(contour_size))]

    # Finds centroid for only the largest contour, the phantom's edge
    M = cv.moments(phantom_contour)
    cx_phantom = int(M['m10'] / M['m00'])
    cy_phantom = int(M['m01'] / M['m00'])
    drawing_image = np.zeros_like(mod_1)
    cv.drawContours(drawing_image, phantom_contour, -1, 255, 1)
    cv.circle(drawing_image, (cx_phantom, cy_phantom), 3, 255, -1)
    row = drawing_image[cy_phantom, :].flatten()
    col = drawing_image[:, cx_phantom].flatten()
    vert_bounds = np.nonzero(col)
    horiz_bounds = np.nonzero(row)
    radius = int((vert_bounds[0][-1] - vert_bounds[0][0]) / 2)
    cy_phantom = vert_bounds[0][0] + radius
    cx_phantom = horiz_bounds[0][0] + radius
    return cx_phantom, cy_phantom

def sample_inserts(module_1,cx,dx,cy,dy,pixel_size):
    contour_list = []
    insert_ctrs = []

    contour_list.append(add_ROI(module_1, cx, dx, cy, dy, pixel_size))
    insert_ctrs.append(len(contour_list) - 1)
    contour_list.append(add_ROI(module_1, cx, dx, cy, -dy, pixel_size))
    insert_ctrs.append(len(contour_list) - 1)
    contour_list.append(add_ROI(module_1, cx, -dx, cy, dy, pixel_size))
    insert_ctrs.append(len(contour_list) - 1)
    contour_list.append(add_ROI(module_1, cx, -dx, cy, -dy, pixel_size))
    insert_ctrs.append(len(contour_list) - 1)
    contour_list.append(add_ROI(module_1, cx, -60, cy, 0, pixel_size))
    insert_ctrs.append(len(contour_list) - 1)

    # should make an insert class and instantiate objects
    bone = []
    air = []
    acrylic = []
    polyethylene = []
    water = []
    unknown = []
    module_1_Dislplay = window_image(module_1.copy(), 0, 400)
    for i in insert_ctrs:
        mask = np.zeros(module_1.shape, np.uint8)
        cv.drawContours(mask, contour_list, i, 255, -1)

        ROI_val = cv.mean(module_1, mask)[0]
        if ROI_val >= 750:  # bone
            bone.append(ROI_val)
            if ROI_val > 850 and ROI_val < 970:
                bone.append(1)
            else:
                bone.append(0)
        elif ROI_val <= -750:  # air
            air.append(ROI_val)
            if ROI_val > -1005 and ROI_val < -970:
                air.append(1)
            else:
                air.append(0)
        elif ROI_val > 50 and ROI_val < 300:  # acrylic
            acrylic.append(ROI_val)
            if ROI_val > 110 and ROI_val < 135:
                acrylic.append(1)
            else:
                acrylic.append(0)
        elif ROI_val > -300 and ROI_val < -50:  # polyethylene
            polyethylene.append(ROI_val)
            if ROI_val > -107 and ROI_val < -84:
                polyethylene.append(1)
            else:
                polyethylene.append(0)
        elif ROI_val > -50 and ROI_val < 50:  # water
            water.append(ROI_val)
            if ROI_val > -7 and ROI_val < 7:
                water.append(1)
            else:
                water.append(0)
        else:  # error, check ROI positioning/scan protocol
            print("ROI Tool failed to detect material")
            print(ROI_val)
            unknown.append(ROI_val)

        contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(module_1_Dislplay, contours, -1, 255, 3)
    cv.circle(module_1_Dislplay, (cx, cy), 3, 255, -1)
    plt.imshow(module_1_Dislplay, cmap='gray')
    plt.show()
    return bone, air, acrylic, polyethylene, water, unknown