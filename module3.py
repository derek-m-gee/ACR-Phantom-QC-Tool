# Module 3
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from imageTools import img_threshold, window_image, add_ROI


#Finds slices with max intensity >150 HU but no more than 800
# Returns slice with the fewest number of non_zero

def find_slice(stack, min_val=300, max_val=800,thresh_level=.6):
    possible_slice = []
    n_nonzero = []
    slice_max = []
    for s in np.arange(len(stack)):
        mask = img_threshold(stack[s],thresh_level)
        if np.count_nonzero(mask > min_val) > 0:
            if mask.max() < max_val:
                possible_slice.append(s)
                n_nonzero.append(np.count_nonzero(mask))
                slice_max.append(mask.max())
    probable_slice = possible_slice[n_nonzero.index(min(n_nonzero))]
    max_HU = slice_max[n_nonzero.index(min(n_nonzero))]
    if (probable_slice - 1 in possible_slice):
        check_max = stack[probable_slice - 1].max()
        if ((check_max - max_HU) / max_HU < .1):
            probable_slice -= 1
    if (probable_slice + 1 in possible_slice):
        check_max = stack[probable_slice + 1].max()
        if ((check_max - max_HU) / max_HU < .1):
            probable_slice += 1
    return probable_slice

def get_BBs(mod_3,level = .6):
    module_3bbs = img_threshold(mod_3.copy(), level)
    mod_3bb_8bit = cv.convertScaleAbs(module_3bbs, alpha=(255 / 32768))
    bb_contours, hierarchy = cv.findContours(mod_3bb_8bit, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #
    # drawing_image = np.zeros_like(mod_3)
    # cv.drawContours(drawing_image, bb_contours, -1, 255, 3)
    return bb_contours

def find_rotation(BB_contours, theta='45'):
    BBx = []
    BBy = []
    for i in np.arange(len(BB_contours)):
        M = cv.moments(BB_contours[i])
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        BBx.append(cx)
        BBy.append(cy)
    BB_centers = list(zip(BBx, BBy))
    p1 = BB_centers[0]
    p2 = BB_centers[1]
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    ang = np.rad2deg(np.arctan2(dy, dx))
    rotation = theta + ang
    return rotation


def test_uniformity(module_3, cx, cy, pixel_size):
    contour_list = []
    insert_ctrs = []
    contour_list.append(add_ROI(module_3, cx, 0, cy, 0, pixel_size, 400))
    insert_ctrs.append(len(contour_list) - 1)
    contour_list.append(add_ROI(module_3, cx, 0, cy, 68, pixel_size, 400))
    insert_ctrs.append(len(contour_list) - 1)
    contour_list.append(add_ROI(module_3, cx, 0, cy, -68, pixel_size, 400))
    insert_ctrs.append(len(contour_list) - 1)
    contour_list.append(add_ROI(module_3, cx, 68, cy, 0, pixel_size, 400))
    insert_ctrs.append(len(contour_list) - 1)
    contour_list.append(add_ROI(module_3, cx, -68, cy, 0, pixel_size, 400))
    insert_ctrs.append(len(contour_list) - 1)

    ROI_val = []
    module_3_Display = module_3.copy()
    module_3_Display = window_image(module_3_Display, 0, 100)
    for i in insert_ctrs:
        mask = np.zeros(module_3.shape, np.uint8)
        cv.drawContours(mask, contour_list, i, 255, -1)
        mean, std_dev = cv.meanStdDev(module_3, mask=mask)
        ROI_val.append([mean.item(), std_dev.item()])
        contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(module_3_Display, contours, -1, 255, 3)
    center = ROI_val[0]
    diff = []
    test_uniformity = []
    for pos in np.arange(1, len(ROI_val)):
        diff.append(ROI_val[pos][0] - ROI_val[0][0])
        if diff[len(diff) - 1] > 7 or diff[len(diff) - 1] < -7:
            test_uniformity.append('FAIL')
        elif diff[len(diff) - 1] > 5 or diff[len(diff) - 1] < -5:
            test_uniformity.append('MINOR DEFICIENCY')
        else:
            test_uniformity.append('PASS')

    # plt.imshow(module_3_Display, cmap='gray')
    # plt.title('12: {}  3: {}  6: {}  9: {}'
    #           '\n Center:{}'
    #           '\nDifference from Center {} {} {} {}'
    #           '\nUniformity within +/- 7 {} {} {} {}'.format(round(ROI_val[1][0], 3), round(ROI_val[3][0], 3),
    #                                                          round(ROI_val[2][0], 3),
    #                                                          round(ROI_val[4][0], 3), round(ROI_val[0][0], 3),
    #                                                          round(diff[0], 2), round(diff[2], 2), round(diff[1], 2),
    #                                                          round(diff[3], 2),
    #                                                          test_uniformity[0], test_uniformity[2], test_uniformity[1],
    #                                                          test_uniformity[3]))
    # plt.show()
    return test_uniformity, module_3_Display
