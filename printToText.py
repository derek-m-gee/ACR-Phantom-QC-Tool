
def createReadMe():
    f= open('FormattingGuideForReportMapping.txt','w')

    f.write('Automated CT ACR QC Reporting \n '
            'Mapping file should be named \"reportMapping\"\n'
            'Each row corresponds to the measurement results for that module\n'
            'The measured values will be reported in the following pattern:\n'
            'Module 1: Water, Air, Bone,  Polyethylene, Acrylic\n'
            'Module 2: mean insert ROI, mean bkg ROI, std dev bkg\n'
            'Module 3: Center ROI, 3 o\'clock ROI, 6 o\'clock ROI \n'
            '9 o\'clock ROI, 12 o\'clock ROI\n'
            'Module 4: # groups\n')
def printToFile(list):

    f = open('test.txt','a')
    for t in list:
        # if isinstance(t,float):
        #     t = str(t)
        line = ' '.join(str(s) for s in t)
        f.write(line + '\n')
    f.write('\n')
