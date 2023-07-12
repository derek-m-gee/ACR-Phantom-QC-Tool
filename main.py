# Automated tool for ACR CT Image Quality QC
#
# References:
# ACR 2017 CT Quality Control Manual
# https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/
# https://www.researchgate.net/publication/9063443_Accuracy_of_a_simple_method_for_deriving_the_presampled_modulation_transfer_function_of_a_digital_radiographic_system_from_an_edge_image
# Create a data/code folder and output

from acrQC import ImageQuality
# Builds QC scan object and scores image quality with respect to ACR Requirements
def main(data_path):
    acr_test = ImageQuality(data_path)
    acr_test.moduleIndexing()
    acr_test.phantomTiltCorrection()
    # CT number accuracy reports Bone, Air, Acrylic, Poly, Water and Pass/Fail(1/0)
    # Pass/Fail determined by ACR requirements for CT number calibration criteria
    acr_test.sampleAccuracyInsert()
    # CNR Pass/Fail based on different limits. Adult (>1) Ped Abd (>0.4) Ped Head (>0.7)
    acr_test.getCNR()
    # Uniformity Pass/Fail determined by measured difference at periphery to center. Difference must be less than 5 HU.
    acr_test.testUniformity()
    # Resolution reported as LP/mm and %difference peak-to-peak
    acr_test.testRes()

    results = [acr_test.inserts,acr_test.CNR,acr_test.uniformity,acr_test.max_res]

    return results

if __name__ == '__main__':
    id = 0
    data_path = r"C:\Users\derek\Documents\RAMD5394IndependentStudy\ACRQCtool\ADULT_ABD_10\\"
    # data_path = r"C:\Users\derek\Documents\RAMD5394IndependentStudy\ACRQCtool\ADULT_ABD_12\\"
    # data_path = r"C:\Users\derek\Documents\RAMD5394IndependentStudy\ACRQCtool\80_kVp_3_mm_4\\"
    # data_path = r"C:\Users\derek\Documents\RAMD5394IndependentStudy\ACRQCtool\PED_BODY_6\\"
    output_path = working_path = r"C:\Users\derek\Documents\RAMD5394IndependentStudy\Test_Project\\"

    test_results = main(data_path)

    pass
