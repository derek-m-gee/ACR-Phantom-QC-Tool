import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from imageTools import window_image, add_ROI
def get_CNR(img_stack, module_2,module_2_index,cx,cy,pixel_size, scan_type = 'adult'):
    contour_list = []
    insert_ctrs = []
    contour_list.append(add_ROI(module_2,cx,0,cy,58,pixel_size,100))
    insert_ctrs.append(len(contour_list) - 1)
    contour_list.append(add_ROI(module_2,cx,-22,cy, 58,pixel_size,100))
    insert_ctrs.append(len(contour_list) - 1)

    # module_2_index = int((module_3_pos - 40)/patient[0].SliceThickness)
    max_CNR = 0
    j = 1
    # plt.figure()
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
        # plt.subplot(2,6, j)
        # plt.imshow(module_2_Dislplay, cmap='gray')
        # plt.title('Slice: {}'.format(test_slice))
        # plt.axis('off')
        j=j+1
        CNR = abs(ROI_val[0][0]-ROI_val[1][0])/ROI_val[1][1]
        if CNR > max_CNR:
            max_CNR = CNR
            module_2_index = test_slice
            module_2_CNR_disp = module_2_Dislplay.copy()
    # plt.show()

    if max_CNR > 1 and scan_type == 'adult':
        test_pass = 'PASS'
    elif max_CNR > 0.4 and scan_type == "ped_abd":
        test_pass = 'PASS'
    elif max_CNR > 0.7 and scan_type == 'ped_head':
        test_pass = 'PASS'
    else:
        test_pass = 'FAIL'

    plt.imshow(module_2_CNR_disp, cmap='gray')
    plt.title('Slice: {}  CNR:{}  {}'.format(module_2_index,round(max_CNR,4),test_pass))
    plt.axis('off')
    plt.show()
    return max_CNR