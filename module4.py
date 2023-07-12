import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from imageTools import window_image

def test_res(module_4,cx,cy,pixel_size):
    rotation = 45
    rotation_matrix = cv.getRotationMatrix2D((int(cx), int(cy)), rotation, 1)
    module_4_Dislplay = module_4.copy()
    height, width = module_4.shape[:2]
    module_4_Dislplay = cv.warpAffine(module_4_Dislplay, rotation_matrix, (width, height))

    module_4_Dislplay = window_image(module_4_Dislplay, 100, 3000)
    ROI_length = 20
    # Ordered pairs indicating locations for resolution patterns
    dx = [-195, -147, -27, 96, 145]
    dy = [-2, 119, 169, 119, -2]
    line_profile = []
    min_val = []
    max_val = []
    deviation = []
    limit = 0.05
    for pattern in np.arange(len(dx)):
        x, y, w, h = int(cx + dx[pattern]), int(
            cy + dy[pattern] - ROI_length / pixel_size[0] * .5), int(ROI_length / pixel_size[0]), int(
            ROI_length / pixel_size[1])
        cv.rectangle(module_4_Dislplay, (x, y), (x + w, y + h), 255, 2)
        roi = module_4_Dislplay[y:y + h, x:x + w]
        line_profile.append(roi.mean(axis=1))
        # Samples central values to avoid non-uniformities at peripheral edges
        min_val.append(min(line_profile[pattern][20:30]))
        max_val.append(max(line_profile[pattern][20:30]))
        deviation.append((max_val[pattern] - min_val[pattern]) / max_val[pattern])
    non_alias_patterns = np.where(np.array(deviation) > limit)[0]
    arr = [deviation[i] for i in non_alias_patterns]
    res_limit_pattern = np.where(arr == np.min(arr))[0]
    LP_pattern = [4, 5, 6, 7, 8, 9, 10, 12]
    res_limit_dev = deviation[int(res_limit_pattern)]

    # rotation_matrix = cv.getRotationMatrix2D((int(cx), int(cy)), -rotation, 1)
    # module_4_Dislplay = cv.warpAffine(module_4_Dislplay, rotation_matrix, (width, height))

    # plt.imshow(module_4_Dislplay, cmap='gray')
    # plt.show()


    # plt.plot(line_profile[int(res_limit_pattern)])
    # plt.xlabel('Pixel distance')
    # plt.ylabel('Intensity')
    # plt.title('Mean Intensity Across Sampled ROI')
    # plt.show()

    return LP_pattern[int(res_limit_pattern)],res_limit_dev, module_4_Dislplay
