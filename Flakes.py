import sys
'''
camera dlls from The Imaging Source calls CoInitialize(Ex) without
a paired CoUninitialize. The thread's COM reference count can' be incremented and stays at 1.
Causing COM error 0x80010106 RPC_E_CHANGED_MODE in multithread situations.
'''
import pythoncom # pip install pypiwin32
pythoncom.CoInitialize()
import warnings
import os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap,QIcon
from PyQt5.QtCore import Qt
import BayerAcq, Histogram, Contour, Rectangle
import ctypes as C
import datetime
import cv2
import subprocess
import time
import signal
from multiprocessing import Process
import msvcrt
import matplotlib.pyplot as plt


'''
Locates and evaluates flakes in microscopic images. 
Citations can be found at the end of this script.

Last edited: August 2019 @anjun
'''


        
    
class Flakes(QWidget):

            
    def __init__(self,parent=None):
        super(Flakes, self).__init__(parent)
        self.minX = 1
        self.minY = 1
        self.maxX = 1
        self.maxY = 1
        
        self.grid = QGridLayout()
        self.resize(600, 750)  
        # file I/O
        now = str(datetime.datetime.today()).replace(":", ".")
        self.default_dir = os.getcwd()
        self.result_path = os.path.join(self.default_dir,str(now))
        self.source_path = os.path.join(self.result_path,'1_Image_Acquisition')
        self.AcqObj = None
        self.HistogramObj = None
        self.ContourObj = None
        self.ROIObj = None
        
        self.acquire_new = True
        self.analysis_bool = True
        self.fine_locate_bool = False
        self.obj_mag = 50
        self.filter = 542
        self.material = 'C'
        
        self.tutorial = None
        self.presentation = 0
        
        print ('Source directory is defaulted to be: ' + self.source_path)
        print ('Output directory is defaulted to be: ' + self.result_path)
        
        self.setWindowIcon(QIcon('icon.ico'))
        self.layout()
        self.setWindowTitle("Flakes")
        self.selectRstDirField.setText(str(self.result_path))
        #self.selectSrcDirField.setText(str(self.source_path))
        self.server = subprocess.Popen('mkdocs serve',shell=False)
        time.sleep(2)
        self.webpage = subprocess.Popen('python -m webbrowser -t "http://127.0.0.1:8000"',shell=False)
        self.SIGINT = False
        self.p = None
        
    def __del__(self):
        print ("Bye!")
        
    
        
        
    def layout(self):
        self.vertical_layout = QVBoxLayout(self)
        self.vertical_layout.setObjectName("vertical_layout")
        self.vertical_layout.addLayout(self.grid)
        spacer_item = QSpacerItem(40, 183, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.vertical_layout.addItem(spacer_item)
        self.setLayout(self.vertical_layout)        
        
        # to acquire or not
        self.acquireCheckBox = QCheckBox("Acquire new images from samples, or...",self)
        self.acquireCheckBox.setChecked(True)
        self.acquireCheckBox.stateChanged.connect(self.toAcquireOrNot)
        self.grid.addWidget(self.acquireCheckBox, 0, 0, 1, 2)
        
        # Select path to source images if not acquiring
        self.selectSrcDirButton=QPushButton(self)
        self.selectSrcDirButton.setObjectName("selectSrcDirDirButton")
        self.selectSrcDirButton.setText('select source directory')
        self.proj_path = self.selectSrcDirButton.clicked.connect(self.selectSrcDirectory)
        self.grid.addWidget(self.selectSrcDirButton, 0,2, 1, 2)
        self.selectSrcDirButton.setEnabled(not self.acquire_new)
        
        self.selectSrcDirField=QLabel(self)
        self.selectSrcDirField.setObjectName("selectSrcDirField")
        self.grid.addWidget(self.selectSrcDirField, 1, 0, 1, 4)
        self.selectSrcDirField.hide()
        
        # choose x and y size
        self.minXField=QLabel(self)
        self.minXField.setText('From min X = ')
        self.grid.addWidget(self.minXField, 2, 0, alignment=Qt.AlignRight)
        self.minYField=QLabel(self)
        self.minYField.setText('min Y = ')
        self.grid.addWidget(self.minYField, 2, 2, alignment=Qt.AlignRight)
        self.maxXField=QLabel(self)
        self.maxXField.setText('To max X = ')
        self.grid.addWidget(self.maxXField, 3, 0, alignment=Qt.AlignRight)
        self.maxYField=QLabel(self)
        self.maxYField.setText('max Y = ')
        self.grid.addWidget(self.maxYField, 3, 2, alignment=Qt.AlignRight)        
        
        self.spX = QSpinBox()
        self.spX.setRange(1,8)
        self.grid.addWidget(self.spX, 2,1)
        self.spY = QSpinBox()
        self.spY.setRange(1,8)
        self.grid.addWidget(self.spY,2,3)
        self.spX2 = QSpinBox()
        self.spX2.setRange(1,8)
        self.grid.addWidget(self.spX2, 3,1)
        self.spY2 = QSpinBox()
        self.spY2.setRange(1,8)
        self.grid.addWidget(self.spY2,3,3)
        self.spinboxset = [self.spX, self.spX2, self.spY, self.spY2]
        for spinbox in self.spinboxset:
            spinbox.setFixedWidth(100)
            spinbox.valueChanged.connect(self.valuechangeSpinbox)
            

        
        
        
        header = QLabel("Objective magnification:")
        self.grid.addWidget(header, 4, 0)  
        self.cbb=QComboBox(self)
        self.cbb.addItem('50')
        #self.cbb.addItem('10')
        self.cbb.setFixedWidth(100)
        self.cbb.currentIndexChanged.connect(self.objSelection)
        self.grid.addWidget(self.cbb, 4, 1) 
        
        
        
        # current narrowband filters: 542, 610, 673 (cannot use infrared w/Nikon)
        self.selectfilterField=QLabel(self)
        self.selectfilterField.setText('Filter wavelength (nm):')
        self.grid.addWidget(self.selectfilterField, 8, 0)
        self.filtercbb=QComboBox(self)
        
        self.filtercbb.addItem('542')
        self.filtercbb.addItem('610')
        self.filtercbb.addItem('ND')
        self.filtercbb.setFixedWidth(100)
        self.filtercbb.currentIndexChanged.connect(self.filterSelection)
        self.grid.addWidget(self.filtercbb, 8, 1) 


        # current narrowband filters: 542, 610, 673 (cannot use infrared w/Nikon)
        self.selectMaterialField=QLabel(self)
        self.selectMaterialField.setText('Material:')
        self.grid.addWidget(self.selectMaterialField, 9, 0)
        self.materialcbb=QComboBox(self)
        self.materialcbb.addItem('C')
        self.materialcbb.addItem('Sb')
        
        self.materialcbb.setFixedWidth(100)
        self.materialcbb.currentIndexChanged.connect(self.materialSelection)
        self.grid.addWidget(self.materialcbb, 9, 1) 
        
        sample_placement_instructions = '''
        

To abort: bring command window to the front and hit Ctrl+C *once*.    
   
---------------------------------------------------------------------------------

Please refer to the pop-up webpage for a complete user guide.
Refresh it if it does not wake up immediately.
 
A few more pop-up windows will jump out to guide you through set-up.


Once you've pressed the Go button the user guide server will go back to sleep, 
so you are not supposed to refresh it anymore.

Thank you for your patience! And sorry about the chirping noise I\'m about to make. 

Have fun!
        

        \n


        
        
        '''
        self.instructionsLabel=QLabel(self)
        self.instructionsLabel.setText(sample_placement_instructions)
        self.grid.addWidget(self.instructionsLabel, 16, 0, 1, 4, alignment=Qt.AlignCenter)
        
        # Select directory for the results
        selectRstDirButton=QPushButton(self)
        selectRstDirButton.setObjectName("selectRstDirButton")
        selectRstDirButton.setText('select result directory')
        self.proj_path = selectRstDirButton.clicked.connect(self.selectRstDirectory)
        self.grid.addWidget(selectRstDirButton, 18,2, 1, 2, alignment=Qt.AlignBottom)
        
        selectRstDirLbl = QLabel("Use default output directory with timestamp or...")
        self.grid.addWidget(selectRstDirLbl, 18, 0, 1, 2) 
        
        self.selectRstDirField=QLabel(self)
        self.selectRstDirField.setObjectName("selectRstDirField")
        self.grid.addWidget(self.selectRstDirField, 19, 0, 1, 2)
        
        self.present_cb = QCheckBox("Presentation Mode")
        self.grid.addWidget(self.present_cb, 10, 0)
        self.present_cb.setChecked(False)
        self.present_cb.stateChanged.connect(self.presentationModeToggle)
        
        demoLbl = QLabel("For new users: choose your favourite material and filter to see the workflow")
        self.grid.addWidget(demoLbl, 11, 0,1,3) 
        demoButton=QPushButton(self)
        demoButton.setText('Demo')
        self.grid.addWidget(demoButton, 11, 3)
        demoButton.setFixedWidth(100)
        demoButton.clicked.connect(self.Demo)
        
        goButton=QPushButton(self)
        goButton.setObjectName("goButton")
        goButton.setText('Go!')
        self.proj_path = goButton.clicked.connect(self.Workflow)
        self.grid.addWidget(goButton, 30, 0, 1, 4)
        
           
    def presentationModeToggle(self):
        present = self.present_cb.isChecked()
        print('presentataion mode is set to '+str(present))
        if present:
            self.presentation = 1
        else: 
            self.presentation = 0
            
    def Demo(self):
        demo_directory = str(self.material)+str(self.filter)
        self.source_path = os.path.join('Demo', demo_directory)
        self.acquire_new = False
        self.presentation = 1
        self.Workflow()
        self.toAcquireOrNot()
        self.presentationModeToggle()
        
    def Workflow (self):
        self.webpage.kill()
        self.server.kill()
        
        now = str(datetime.datetime.today()).replace(":", ".")
        self.result_path = os.path.join(self.default_dir,str(now))
        self.selectRstDirField.setText(self.result_path)
        print ('Output directory is chosen to be: ' + self.result_path)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        
        stats_file = open(os.path.join(self.result_path, 'ListOfFlakes.csv') ,"a+")
        stat = "LithoCoord,Contrast,BoundingBoxArea(um^2),EstLayers\n"
        stats_file.write(stat)
        stats_file.close()
        
        #signal.signal(signal.SIGINT, self.signal_handler)
        
        try:
            if self.acquire_new:
                #stage_check_popup = StageCheck()
                #C.windll.user32.MessageBoxW(0, "All 3 axes (X, Y and Z) should be positioned (approximately) at the midpoint of their moving ranges as shown in the picture.\nAdjust them with a #3 hex wrench if needed.", "Stage Check", 0)
                C.windll.user32.MessageBoxW(0, "Please turn on the picomotor controller. \nMake sure that the picomotor gears are not touching anything \n(i.e. their own wires, microscope objectives, etc.)", "Motor Check", 0)
                C.windll.user32.MessageBoxW(0, "Insert your filter. Take a look at the binoculars, do you see your desired colour? \n\nMove the stages around. Keeping the grids orthogonal. \nMove until you can see your starting coordinate numbers \n(the ones you specified as min X and min Y) at the centre of the field.", "Alignment", 0)
                C.windll.user32.MessageBoxW(0, "FOCUS on that area and pull out the BINO-PHOTO slider.\nWe will move to the screen.", "Almost there!", 0)
                self.Acquire()
            else:
                self.Analyse()
        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            plt.close("all")
            #C.windll.user32.MessageBoxW(0, "I couldn't find the images...\nSource directory should contain images (and images only!)", "Check source directory", 0)
            print ('Abort')
            return
            
        
            
    def signal_handler(self,signal, frame):
        self.p.terminate() 
        self.p.join()


    def Acquire (self):
        self.AcqObj = BayerAcq.BayerAcq(self.result_path, minX=self.minX, minY=self.minY, maxX=self.maxX, maxY=self.maxY, ftr = self.filter, objective_magnification=self.obj_mag, material = self.material,presentation = self.presentation)
        self.source_path = os.path.join(self.result_path,'1_Image_Acquisition')
        print ('Source directory is chosen to be: ' + self.source_path)
        return
        
        
    def Analyse (self):   
            steps = ['', '', '1_Image_Acquisition','Global_Histogram','2_Segmentation', '3_ROIs', '4_Local_Histogram']
            full_path = []
            for step in steps:
                full_path.append(os.path.join(self.result_path,step))
            #print (full_path)
            if not self.acquire_new:
                full_path[2] = self.source_path
            list_of_samples = os.listdir(full_path[2])
            
            stats_file = open(os.path.join(self.result_path, 'ListOfFlakes.csv') ,"a+")
            stat = "LithoCoord,Contrast,EstLayers,BoundingBoxArea(um^2)\n"
            stats_file.write(stat)
            stats_file.close()
            
            for sample in list_of_samples:
                #if msvcrt.kbhit() and msvcrt.getch()=='q':
                #    print('Quit')
                #    return
                #sample_path = os.path.join(full_path[2], str(sample))
                print(sample)
                self.ContourObj = Contour.Contour(str(sample), full_path[2], self.result_path, obj_mag=self.obj_mag, material = self.material,ftr=self.filter)
                if self.material == 'Sb':
                    self.ContourObj.edges_gluey(obj_mag=self.obj_mag, presentation=self.presentation)
                else:
                    self.ContourObj.edges_gluey(obj_mag=self.obj_mag, ftr=self.filter, glue_remover=False, presentation=self.presentation)
                self.ROIObj = Rectangle.Rectangle((full_path[4]+'\\Contours'), sample, full_path[2], self.result_path, obj_mag=self.obj_mag, material=self.material,ftr=self.filter)
                X_Y_FS = self.ROIObj.markROIs(save=True, presentation=self.presentation)
                
                list_of_chunks = os.listdir((full_path[5]+'\\'+sample[:-4]))
                list_of_chunk_numbers = []
                if self.material == 'C':  
                    for chunk,ant in zip(list_of_chunks,X_Y_FS):
                        stat = str(sample)[:-4]+'_'
                        self.HistogramObj = Histogram.Histogram(str(chunk), (full_path[5]+'\\'+sample[:-4]), (full_path[6]), self.result_path,obj_mag=self.obj_mag)
                        #self.HistogramObj = Histogram.Histogram(str(chunk), (full_path[5]+'\\'+sample), (full_path[6]+'\\'+sample), self.result_path,obj_mag=self.obj_mag) #deeper directory
                        contrast,layers = self.HistogramObj.saveLocalHistogram(ftr=self.filter, material=self.material, presentation=self.presentation)
                        if not contrast is None and len(contrast)>3:
                            chunk_number = int(chunk[-6:-4])
                            stat += str(chunk_number).zfill(2) + ',' + contrast
                            stat +=  ',' + layers
                            list_of_chunk_numbers.append(chunk_number)
                        if ant[3] in list_of_chunk_numbers:
                            stat += ","+str(ant[2]) +"\n"
                            stats_file = open(os.path.join(self.result_path, 'ListOfFlakes.csv') ,"a+")
                            stats_file.write(stat)
                            stats_file.close()
                
                cv2.destroyAllWindows()
                plt.close("all")
                
                
 
    def FineLocate (self):
        pass
    
    def valuechangeSpinbox (self):
        self.minX = self.spX.value()
        if self.minX>self.maxX:
            self.maxX = self.minX
            self.spX2.setValue(self.maxX)
        self.minY = self.spY.value()
        if self.minY>self.maxY:
            self.maxY = self.minY
            self.spY2.setValue(self.maxY)
        self.maxX = self.spX2.value()
        if self.minX>self.maxX:
            self.minX = self.maxX
            self.spX.setValue(self.minX)
        self.maxY = self.spY2.value()
        if self.minY>self.maxY:
            self.minY = self.maxY
            self.spY.setValue(self.minY)
        print("Grid area to be scanned: from "+str(self.minX)+str(self.minY)+' to '+str(self.maxX)+str(self.maxY))
        
    def toAcquireOrNot (self):
        if self.acquireCheckBox.isChecked():
            self.acquire_new = True
            self.selectSrcDirField.setText('')
            self.selectSrcDirField.hide()
            self.spX.show()
            self.spY.show()
            self.spX2.show()
            self.spY2.show()
            self.minXField.show()
            self.minYField.show()
            self.maxXField.show()
            self.maxYField.show()
        else:
            self.acquire_new = False
            self.selectSrcDirField.show()
            self.selectSrcDirField.setText(str(self.source_path))
            self.spX.hide()
            self.spY.hide()
            self.spX2.hide()
            self.spY2.hide()
            self.minXField.hide()
            self.minYField.hide()
            self.maxXField.hide()
            self.maxYField.hide()
        print ('Acquire new samples? ' + str(self.acquire_new) )
        self.selectSrcDirButton.setEnabled(not self.acquire_new)
        return self.acquire_new
    
    def toAnalyseOrNot (self, state):
        if self.analyseCheckBox.isChecked():
            self.analysis_bool = True
        else:
            self.analysis_bool = False
        print ('Analyse? ' + str(self.analysis_bool) )
        return self.analysis_bool


    def objSelection (self):
        self.obj_mag = int(self.cbb.currentText())
        print ('Objective? ' + str(self.obj_mag) )
        return self.obj_mag

    
    def filterSelection (self):
        try:
            self.filter = int(self.filtercbb.currentText())
            print ('Filter central wavelength (nm): ' + str(self.filter) )
        except:
            self.filter = 0
            print ('Using neutral density filter')
        return self.filter
    
    def materialSelection (self):
        self.material =  self.materialcbb.currentText()
        if self.material == 'Sb':
            #NDindex = self.materialcbb.findText('ND')
            self.filtercbb.setCurrentText('ND')
            self.filtercbb.setEnabled(False)
        else:
            self.filtercbb.setEnabled(True)
        print ('Material: ' + str(self.material) )
        return self.material
    
    def selectSrcDirectory (self):
        proj_path = QFileDialog.getExistingDirectory(self, 'Select Source Directory')
        if ((not proj_path == '') and (not self.acquire_new)) :
            self.selectSrcDirField.setText(str(proj_path))
            self.source_path = str(proj_path)
            print ('Source directory is chosen to be: ' + self.source_path)
            return proj_path
        elif (self.acquire_new):
            self.source_path = self.result_path+'\\1_Image_Acquisition'
        else:
            return self.source_path
    
    
    def selectRstDirectory (self):
        proj_path = QFileDialog.getExistingDirectory(self, 'Select Result Directory')
        if not str(proj_path) == '' :
            self.selectRstDirField.setText(str(proj_path))
            self.result_path = str(proj_path)
            print ('Output directory is chosen to be: ' + self.result_path)
            return proj_path
        else:
            return self.result_path
        
    
    def setParam(self):
        pass            

def Acquire (result_path, minX, minY, maxX, maxY, obj_mag, material, ftr, presentation):
    BayerAcq.BayerAcq(result_path, minX, minY, maxX, maxY, ftr, obj_mag, material, presentation)
    source_path = os.path.join(result_path,'1_Image_Acquisition')
    print ('Source directory is chosen to be: ' + source_path)
    return
        
        
def Analyse (acquire_new, result_path,source_path,obj_mag,material, ftr, presentation):   
            steps = ['', '', '1_Image_Acquisition','Global_Histogram','2_Segmentation', '3_ROIs', '4_Local_Histogram']
            full_path = []
            for step in steps:
                full_path.append(os.path.join(result_path,step))
            #print (full_path)
            if not acquire_new:
                full_path[2] = source_path
            list_of_samples = os.listdir(full_path[2])
            
            stats_file = open(os.path.join(result_path, 'ListOfFlakes.csv') ,"a+")
            stat = "LithoCoord,Contrast,EstLayers,BoundingBoxArea(um^2)\n"
            stats_file.write(stat)
            stats_file.close()
            
            for sample in list_of_samples:
                #sample_path = os.path.join(full_path[2], str(sample))
                print(sample)
                ContourObj = Contour.Contour(str(sample), full_path[2], result_path, obj_mag, material,ftr)
                if material == 'Sb':
                    ContourObj.edges_gluey(obj_mag=obj_mag, ftr=0, glue_remover=True, presentation=presentation)
                else:
                    ContourObj.edges_gluey(obj_mag, ftr, glue_remover=False, presentation=presentation)
                #self.ContourObj.segmentation(ftr=self.filter)
                ROIObj = Rectangle.Rectangle((full_path[4]+'\\Contours'), sample, full_path[2], result_path, obj_mag, material,ftr)
                X_Y_FS = ROIObj.markROIs(True, presentation)
                
                list_of_chunks = os.listdir((full_path[5]+'\\'+sample[:-4]))
                list_of_chunk_numbers = []
                if material == 'C':  
                    for chunk,ant in zip(list_of_chunks,X_Y_FS):
                        stat = str(sample)[:-4]+'_'
                        HistogramObj = Histogram.Histogram(str(chunk), (full_path[5]+'\\'+sample[:-4]), (full_path[6]),result_path,obj_mag)
                        #self.HistogramObj = Histogram.Histogram(str(chunk), (full_path[5]+'\\'+sample), (full_path[6]+'\\'+sample), self.result_path,obj_mag=self.obj_mag) #deeper directory
                        contrast,layers = HistogramObj.saveLocalHistogram(ftr, material, presentation)
                        if not contrast is None and len(contrast)>3:
                            chunk_number = int(chunk[-6:-4])
                            stat += str(chunk_number).zfill(2) + ',' + contrast
                            stat +=  ',' + layers
                            list_of_chunk_numbers.append(chunk_number)
                        if ant[3] in list_of_chunk_numbers:
                            #TB = 'T' if (ant[1]<256) else ('B')
                            #LR = 'L' if (ant[0]<256) else ('R')
                            stat += ","+str(ant[2]) +"\n"
                            stats_file = open(os.path.join(result_path, 'ListOfFlakes.csv') ,"a+")
                            stats_file.write(stat)
                            stats_file.close()
                
                cv2.destroyAllWindows()


class StageCheck (QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Stage Check'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.initStageCheck()
    
    def initStageCheck(self):
        self.setWindowTitle(self.title)
        label = QLabel(self)
        pixmap = QPixmap("LithoExamplar\stagecheck.png")
        label.setPixmap(pixmap)
        self.resize(pixmap.width(),pixmap.height()+10)
        self.show()
        
    
if __name__=="__main__":
    warnings.simplefilter("ignore", UserWarning)
    sys.coinit_flags = 2
    app = QApplication(sys.argv)
    ex = Flakes()
    ex.show()

#    ex.remove_all_lines()
    sys.exit(app.exec_())



"""
Citations:
        
        # Reflection contrast
        Blake, P., Hill, E. W., Castro Neto, A. H., Novoselov, K. S., Jiang, D., Yang, R., ... & Geim, A. K. (2007). Making graphene visible. Applied physics letters, 91(6), 063124.
        Gaskell, P. E., Skulason, H. S., Rodenchuk, C., & Szkopek, T. (2009). Counting graphene layers on glass via optical reflection microscopy. Applied physics letters, 94(14), 143101.
        
        # Autofocusing and edge detection
        Lindeberg, T. (1998). Edge detection and ridge detection with automatic scale selection. International journal of computer vision, 30(2), 117-156.
        Otsu, N. (1979). A threshold selection method from gray-level histograms. IEEE transactions on systems, man, and cybernetics, 9(1), 62-66.
        
        # Curve smoothing
        https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
        
        # Wrapper for IC Imaging Source Camera Driver
        https://github.com/anjunhu/IC-Imaging-Control-Samples
        
        # Wrapper for Picomotor Controller/Driver
        https://github.com/bdhammel/python_newport_controller
        
"""