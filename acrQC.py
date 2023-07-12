from imageTools import *
from module1 import *
from module2 import *
from module3 import *
from module4 import *
class ImageQuality:
    def __init__(self, data_path):
        self.file_path = data_path
        self.raw_data = load_scan(data_path)
        self.CT_data = get_pixels_hu_stack(self.raw_data)
        self.pixel_size = self.raw_data[0].PixelSpacing
        self.kVP = self.raw_data[0].KVP

        if self.kVP <= 100:
            self.min_level = 200
        elif self.kVP == 120:
            self.min_level = 300

    def moduleIndexing(self):
        #######################################################################################
        # Module 1 Indexing
        # Uses Module 3 BBs as landmark to determine Module 1 slice position.
        # ACR Phantom Modules 40 mm in depth and 200 mm diameter.
        # subtracting the length of 2 modules from the position of the
        # known module 3 slice position gives the position of module 1.
        # Check if module 1 is in the correct direction use air pixels to confirm, can check both directions should have an air and phantom measurement
        # Air air - wrong direction
        # air phantom - correct direction
        #######################################################################################
        self.module_3_index = find_slice(self.CT_data,self.min_level)
        self.module_3 = self.CT_data[self.module_3_index]
        self.module_3_pos = self.module_3_index * self.raw_data[0].SliceThickness  # mm
        try:
            self.module_1_index = int((self.module_3_pos - 80) / self.raw_data[0].SliceThickness)
            self.module_1 = self.CT_data[self.module_1_index]
            mask = np.zeros(self.module_1.shape, np.uint8)
            mod_1_8bit = cv.convertScaleAbs(self.module_1, alpha=(255 / 32768))
            contours, hierarchy = cv.findContours(mod_1_8bit, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            cv.drawContours(mask, contours, -1, 255, -1)

            mean_val = cv.mean(self.module_1, mask)[0]
            if mean_val < -500:
                raise IndexError
        except IndexError:
            # print("Error Phantom Rotated")
            np.flip(self.CT_data)
            self.module_3_index = find_slice(self.CT_data, self.min_level)
            self.module_3 = self.CT_data[self.module_3_index]
            self.module_3_pos = self.module_3_index * self.raw_data[0].SliceThickness
            self.module_1_index = int((self.module_3_pos - 80) / self.raw_data[0].SliceThickness)
            self.module_1 = self.CT_data[self.module_1_index]
        try:
            self.module_4_index = int((self.module_3_pos + 40) / self.raw_data[0].SliceThickness)
            self.module_4 = self.CT_data[self.module_4_index]
        except IndexError:
            print("Error Phantom Rotated")
            np.flip(self.CT_data)
            self.module_3_index = find_slice(self.CT_data, self.min_level)
            self.module_3 = self.CT_data[self.module_3_index]
            self.module_3_pos = self.module_3_index * self.raw_data[0].SliceThickness
            self.module_1_index = int((self.module_3_pos - 80) / self.raw_data[0].SliceThickness)
            self.module_1 = self.CT_data[self.module_1_index]
        self.module_2_index = int((self.module_3_pos - 40) / self.raw_data[0].SliceThickness)
        self.module_2 = self.CT_data[self.module_2_index]
        # if loop over 
        # plt.imshow(module_1, cmap='gray')
        # plt.show()

    def phantomTiltCorrection(self):
        #######################################################################################
        # Phantom Tilt Correction
        #######################################################################################
        BB_contours = get_BBs(self.module_3)
        # drawing_image = np.zeros_like(self.module_3)
        # cv.drawContours(drawing_image,BB_contours,-1,255,3)
        offset = find_rotation(BB_contours, 45)

        self.height, self.width = self.module_3.shape[:2]
        self.rotation_matrix = cv.getRotationMatrix2D((self.width / 2, self.height / 2), offset, 1)
        for i in np.arange(len(self.CT_data)):
            self.CT_data[i] = cv.warpAffine(self.CT_data[i], self.rotation_matrix, (self.width, self.height))

    def sampleAccuracyInsert(self):
        #######################################################################################
        # Module 1: CT Number Accuracy Operations
        # Phantom Edge Detection and Centering
        #######################################################################################

        # needs to be rotated again since this data is from the patient set, not the img_stack
        mod_1_8bit = self.raw_data[self.module_1_index].pixel_array.copy()
        mod_1_8bit = cv.warpAffine(mod_1_8bit, self.rotation_matrix, (self.width, self.height))
        mod_1_8bit = cv.convertScaleAbs(mod_1_8bit, alpha=(255 / 32768))
        # plt.imshow(edges, cmap='gray')
        # plt.show()
        self.cx_phantom, self.cy_phantom = phantom_center(mod_1_8bit)

        # Allows for adjustment of ROI placement
        dx = 44.25
        dy = 43.75
        # Requires module 1 in HU
        self.inserts = sample_inserts(self.module_1, self.cx_phantom, dx, self.cy_phantom, dy, self.pixel_size)

        # self.bone, self.air, self.acrylic, self.polyethylene, self.water, self.unknown = sample_inserts(self.module_1,
        #                                                                                                 self.cx_phantom, dx,
        #                                                                                                 self.cy_phantom, dy,
        #                                                                                                 pixel_size)
    def getCNR(self,scan_type = 'adult'):
        #######################################################################################
        # Module 2: CNR
        #######################################################################################
        self.CNR = get_CNR(self.CT_data, self.module_2, self.module_2_index, self.cx_phantom, self.cy_phantom, self.pixel_size, scan_type)
    def testUniformity(self):
        #######################################################################################
        # Module 3: Uniformity
        #######################################################################################
        self.uniformity = test_uniformity(self.module_3, self.cx_phantom, self.cy_phantom, self.pixel_size)

    def testRes(self):
        #######################################################################################
        # Module 4: Spatial Resolution
        #######################################################################################
        self.max_res = test_res(self.module_4, self.cx_phantom, self.cy_phantom, self.pixel_size)