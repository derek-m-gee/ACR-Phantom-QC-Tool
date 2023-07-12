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
        img_widget = qtw.QLabel("Placeholder for module slice")
        pass_fail = qtw.QLabel("Pass/Fail Indicator")

        self.figure_1 = qtw.QLabel()
        # self.figure_1 = MplCanvas(self,width=5,height=4,dpi=100)
        self.figure_2 = MplCanvas(self,width=5,height=4,dpi=100)
        self.figure_3 = MplCanvas(self,width=5,height=4,dpi=100)
        self.figure_4 = MplCanvas(self,width=5,height=4,dpi=100)
        # self.setCentralWidget(sc)

        tab_widget = qtw.QTabWidget()
        self.setCentralWidget(tab_widget)

        mod1container = qtw.QWidget(self)
        mod2container = qtw.QWidget(self)
        mod3container = qtw.QWidget(self)
        mod4container = qtw.QWidget(self)

        grid_layout = qtw.QGridLayout()

        grid_layout.addWidget(img_widget)
        grid_layout.addWidget(pass_fail)

        #layout.addLayout(grid_layout)
        mod1container.setLayout(qtw.QVBoxLayout())
        mod1container.layout().addWidget(self.figure_1)
        mod1container.layout().addWidget(pass_fail)
        mod2container.setLayout(qtw.QVBoxLayout())
        mod2container.layout().addWidget(self.figure_2)
        mod2container.layout().addWidget(pass_fail)
        mod3container.setLayout(qtw.QVBoxLayout())
        mod3container.layout().addWidget(self.figure_3)
        mod3container.layout().addWidget(pass_fail)
        mod4container.setLayout(qtw.QVBoxLayout())
        mod4container.layout().addWidget(self.figure_4)
        mod4container.layout().addWidget(pass_fail)

        tab_widget.addTab(mod1container, 'Module 1')
        tab_widget.addTab(mod2container, 'Module 2')
        tab_widget.addTab(mod3container, 'Module 3')
        tab_widget.addTab(mod4container, 'Module 4')

        # sc.addTab(mod1container, 'Module 1')
        # sc.addTab(mod2container, 'Module 2')
        # sc.addTab(mod3container, 'Module 3')
        # sc.addTab(mod4container, 'Module 4')

        dock = qtw.QDockWidget("Open")
        self.addDockWidget(qtc.Qt.LeftDockWidgetArea,dock)

        dock.setFeatures(
            qtw.QDockWidget.DockWidgetMovable |
            qtw.QDockWidget.DockWidgetFloatable
        )

        configuration_widget = qtw.QWidget()
        configuration_widget.setLayout(qtw.QVBoxLayout())
        dock.setWidget(configuration_widget)
        self.filepath = ""
        self.file_select = qtw.QLabel('Select a DICOM set')
        self.file_search_btn = qtw.QPushButton(
            "Search",
            clicked=self.selectFile
        )
        self.run_main_btn = qtw.QPushButton(
            "Run QC Protocol",
            clicked=self.runQC
        )
        # self.check_file_path()

        # self.quitbutton = qtw.QPushButton('Quit', clicked=self.close)

        configuration_widget.layout().addWidget(self.file_select)
        configuration_widget.layout().addWidget(self.file_search_btn)
        configuration_widget.layout().addWidget(self.run_main_btn)
        # configuration_widget.layout().addWidget(quitbutton)
        configuration_widget.layout().addStretch()
        # Configure widgets

        # Event Categories

        # Arrange widgets

        main_layout = qtw.QHBoxLayout()
        self.setLayout(main_layout)

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
            results = main(self.filepath)
            fig_mod1 = results[0][6]
            width, height = fig_mod1.shape
            qImg = qtg.QPixmap(qtg.QImage(fig_mod1,width,height,qtg.QImage.Format_Grayscale8))
            self.figure_1.setPixmap(qImg)
            # self.figure_1.axes.plot(results[0][6])
            self.figure_2.axes.plot([0, 1, 2, 3, 4], [10, 1, 20, 3, 40])

            self.figure_2.axes.plot([0, 1, 2, 3, 4], [10, 1, 20, 3, 40])
            self.figure_3.axes.plot([0, 1, 2, 3, 4], [10, 1, 20, 3, 40])
            self.figure_4.axes.plot([0, 1, 2, 3, 4], [10, 1, 20, 3, 40])
        except:
            self.figure_1.setText("error")
    # Interface with objects to open and store data

    # create txt file, use qt to find file, use another python back end, while no exit code then wait
    # pass path name to back end starter file import

    def check_file_path(self):
        self.run_main_btn.setDisabled(
            self.filepath == ""
        )

if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    mw = MainWindow()
    sys.exit(app.exec())