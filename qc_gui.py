import sys
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc
from main import main

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self,parent=None,width=5,height=4,dpi=100):
        fig = Figure(figsize=(width,height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas,self).__init__(fig)

class MainWindow(qtw.QMainWindow):

    def __init__(self):
        """MainWindow constructor"""
        super().__init__()
        # Main UI code goes here
        # Configure window
        self.setWindowTitle("ACR CT Quality Control")
        self.resize(800,600)

        # Create widgets

        # self.textedit = qtw.QTextEdit()
        # self.setCentralWidget(self.textedit)
        pass_fail = qtw.QLabel("Pass/Fail Indicator")

        self.figure_1 = qtw.QLabel("Select a data set")
        self.figure_2 = qtw.QLabel("Select a data set")
        self.figure_3 = qtw.QLabel("Select a data set")
        self.figure_4 = qtw.QLabel("Select a data set")
        self.module1_accuracy = qtw.QLabel("HU Accuracy:")
        self.module1_accuracy.setFixedWidth(600)
        self.module2_cnr = qtw.QLabel("CNR:")
        self.module2_cnr.setFixedWidth(600)
        self.module3_uniformity = qtw.QLabel("Uniformity:")
        self.module3_uniformity.setFixedWidth(600)
        self.module4_res = qtw.QLabel("Maximum Resolution:")
        self.module4_res.setFixedWidth(600)

        main_layout = qtw.QHBoxLayout()
        configuration_widget = qtw.QWidget()
        configuration_widget.setLayout(qtw.QVBoxLayout())
        button_layout = qtw.QHBoxLayout()
        self.b1 = qtw.QRadioButton("Adult")
        self.b1.setChecked(True)
        # self.b1.toggled.connect(lambda:self.btnstate(self.b1))
        button_layout.addWidget(self.b1)

        self.b2 = qtw.QRadioButton("Child")
        # self.b2.toggled.connect(lambda:self.btnstate(self.b2))
        button_layout.addWidget(self.b2)
        scan_type = qtw.QWidget()
        scan_type.setLayout(button_layout)

        main_layout.addWidget(configuration_widget)
        tab_widget = qtw.QTabWidget()
        main_layout.addWidget(tab_widget)

        mod1container = qtw.QWidget(self)
        mod2container = qtw.QWidget(self)
        mod3container = qtw.QWidget(self)
        mod4container = qtw.QWidget(self)

        mod1container.setLayout(qtw.QVBoxLayout())
        mod1container.layout().addWidget(self.figure_1)
        mod1container.layout().addWidget(self.module1_accuracy)

        mod2container.setLayout(qtw.QVBoxLayout())
        mod2container.layout().addWidget(self.figure_2)
        mod2container.layout().addWidget(self.module2_cnr)

        mod3container.setLayout(qtw.QVBoxLayout())
        mod3container.layout().addWidget(self.figure_3)
        mod3container.layout().addWidget(self.module3_uniformity)

        mod4container.setLayout(qtw.QVBoxLayout())
        mod4container.layout().addWidget(self.figure_4)
        mod4container.layout().addWidget(self.module4_res)

        tab_widget.addTab(mod1container, 'Module 1')
        tab_widget.addTab(mod2container, 'Module 2')
        tab_widget.addTab(mod3container, 'Module 3')
        tab_widget.addTab(mod4container, 'Module 4')

        self.filepath = ""
        self.file_select = qtw.QLabel('Select a DICOM set')
        self.file_select.setFixedWidth(125)
        self.file_search_btn = qtw.QPushButton(
            "Search",
            clicked=self.selectFile
        )
        type_selection = qtw.QLabel('Select a Scan Type')
        self.run_main_btn = qtw.QPushButton(
            "Run QC Protocol",
            clicked=self.runQC
        )
        # self.check_file_path()
        # self.quitbutton = qtw.QPushButton('Quit', clicked=self.close)

        configuration_widget.layout().addWidget(self.file_select)
        configuration_widget.layout().addWidget(self.file_search_btn)
        configuration_widget.layout().addWidget(type_selection)
        configuration_widget.layout().addWidget(scan_type)
        configuration_widget.layout().addWidget(self.run_main_btn)
        # configuration_widget.layout().addWidget(quitbutton)
        configuration_widget.layout().addStretch()
        # Configure widgets

        # Event Categories

        # Arrange widgets
        central_widget = qtw.QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        # End Main UI code
        self.show()

    def selectFile(self):
        self.filepath = qtw.QFileDialog.getExistingDirectory(
            self, "Select a Study...",
            qtc.QDir.homePath(),
            qtw.QFileDialog.ShowDirsOnly
        )
        self.file_select.setText(self.filepath)
        # self.run_main_btn.setEnabled()
        # if filename:
        #     try:
        #         with open(filename, 'r') as fh:
        #             self.textedit.setText(fh.read())
        #     except Exception as e:
        #         qtw.QMessageBox.critical(f"Could not load file: {e}")

    def runQC(self):
        try:
            if self.filepath == "":
                self.figure_1.setText("Please Select A File and Try Again")
                self.figure_2.setText("Please Select A File and Try Again")
                self.figure_3.setText("Please Select A File and Try Again")
                self.figure_4.setText("Please Select A File and Try Again")
            else:
                results = main(self.filepath)

                fig_mod1 = qtg.QPixmap('module1.png')
                fig_mod2 = qtg.QPixmap('module2.png')
                fig_mod3 = qtg.QPixmap('module3.png')
                fig_mod4 = qtg.QPixmap('module4.png')
                width = fig_mod1.width()
                height = fig_mod1.height()
                self.figure_1.setPixmap(fig_mod1)
                self.figure_2.setPixmap(fig_mod2)
                self.figure_3.setPixmap(fig_mod3)
                self.figure_4.setPixmap(fig_mod4)


                self.module1_accuracy.setText("HU Accuracy:   Bone:{:.2f}   Air:{:.2f}   Acrylic:{:.2f}   Poly:{:.2f}   Water:{:.2f}"
                                              "\n                       {}            {}            {}                {}            {}"
                                              .format(results[0][0][0],results[0][1][0],
                                                      results[0][2][0],results[0][3][0],
                                                      results[0][4][0],
                                                      results[0][0][1], results[0][1][1],
                                                      results[0][2][1], results[0][3][1],
                                                      results[0][4][1]))

                # self.module1_accuracy.setText("HU Accuracy: Bone:{:.2f} {} Air:{:.2f} {} Acrylic:{:.2f} {}"
                #                               " Poly:{:.2f} {} Water:{:.2f} {}"
                #                               .format(results[0][0][0],results[0][0][1],results[0][1][0],
                #                                       results[0][1][1],results[0][2][0],results[0][2][2],
                #                                       results[0][3][0],results[0][3][1],
                #                                       results[0][4][0],results[0][4][1]))
                self.module2_cnr.setText("CNR Test: {}".format(results[1][3]))
                self.module3_uniformity.setText("Uniformity: {}".format(results[2][2]))
                self.module4_res.setText("Maximum Resolution: {} LP/mm".format(results[3][0]))


            # Create export folder with Excel interface
            # Try to implement color coded tabs for pass/fail
            # Fix string display of file name using negative
            # Bold the correct spatial resolution tool

            # qImg_mod1 = qtg.QPixmap(qtg.QImage(fig_mod1,width,height,qtg.QImage.Format_Grayscale16))
            # self.figure_1.setPixmap(qImg_mod1)
            # qImg_mod2 = qtg.QPixmap(qtg.QImage(fig_mod2,width,height,qtg.QImage.Format_Grayscale16))
            # self.figure_2.setPixmap(qImg_mod2)
            # qImg_mod3 = qtg.QPixmap(qtg.QImage(fig_mod3,width,height,qtg.QImage.Format_Grayscale16))
            # self.figure_3.setPixmap(qImg_mod3)
            # qImg_mod4 = qtg.QPixmap(qtg.QImage(fig_mod4,width,height,qtg.QImage.Format_Grayscale16))
            # self.figure_4.setPixmap(qImg_mod4)

        except:
            self.figure_1.setText("Error, Try Again. Be sure to select a folder"
                                  "with the CT data set")
            self.figure_2.setText("Error, Try Again. Be sure to select a folder"
                                  "with the CT data set")
            self.figure_3.setText("Error, Try Again. Be sure to select a folder"
                                  "with the CT data set")
            self.figure_4.setText("Error, Try Again. Be sure to select a folder"
                                  "with the CT data set")

    def check_file_path(self):
        self.run_main_btn.setDisabled(
            self.filepath == ""
        )

    def btnstate(selfself,b):

        if b.text() == "Adult":
            if b.isChecked() == True:
                print(b.text()+" is selected")
            else:
                print(b.text()+" is deslected")

        if b.text() == "Child":
            if b.isChecked() == True:
                print(b.text()+" is selected")
            else:
                print(b.text()+" is deslected")

if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    mw = MainWindow()
    sys.exit(app.exec())