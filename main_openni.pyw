import pdb
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
import qimage2ndarray
from gui0 import *
import cv2
import sys,time,os,queue
import numpy as np
from kinect_controller import KinectRefresh,KinectScheduler,KinectSave
import pcl
import pcl.pcl_visualization
from openni import openni2
from HPS3D_controller import HPS3DRefresh, HPS3DSave

class RecordWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, depth_stream, parent=None):
        super(RecordWindow, self).__init__(parent)
        self.setupUi(self)
        self.KinectRefresh_thread = KinectRefresh()
        self.KinectRefresh_thread.start()
        self.HPS3DRefresh_thread = HPS3DRefresh(depth_stream)
        self.HPS3DRefresh_thread.start()
        self.depth_stream = depth_stream
        #self.DrawThread_thread = DrawThread()
        #self.DrawThread_thread.start()
        self.isrecord = False
        self.slot_init()

    def slot_init(self):
        self.recordButton.clicked.connect(self.button_record_clicked)
        self.KinectRefresh_thread.newKinectRGBSk.connect(self.processnewKinectRGBSk)
        self.KinectRefresh_thread.newKinectRGB.connect(self.processnewKinectRGB)
        self.KinectRefresh_thread.newKinectSk.connect(self.processnewKinectSk)
        self.KinectRefresh_thread.newKinectDepth.connect(self.processnewKinectDepth)
        self.KinectRefresh_thread.newKinectIR.connect(self.processnewKinectIR)
        self.HPS3DRefresh_thread.newCloud.connect(self.processnewCloud)

    def processnewKinectRGBSk(self,newimg):
        img = cv2.resize(newimg,(512,288))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = qimage2ndarray.array2qimage(img)
        self.rgbLabel.setPixmap(QtGui.QPixmap(qimg))

    def processnewKinectRGB(self,data):
        if self.isrecord:
            self.save_scheduler.update(data,"RGBimage")

    def processnewKinectSk(self,data):
        if self.isrecord:
            self.save_scheduler.update(data,"Skdata")
    
    def processnewCloud(self,data):
        if self.isrecord:
            self.save_scheduler.update(data,"Cloud")

        R = (data//256).astype(np.uint8)
        G = (data%256).astype(np.uint8)
        B = np.zeros_like(R).astype(np.uint8)
        img = np.concatenate([R,G,B], 2)
        qimg = qimage2ndarray.array2qimage(img)
        self.label4.setPixmap(QtGui.QPixmap(qimg))

    def processnewKinectDepth(self,data):
        if self.isrecord:
            self.save_scheduler.update(data,"Depthimage")
        img = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        qimg = qimage2ndarray.array2qimage(img)
        self.depthLabel.setPixmap(QtGui.QPixmap(qimg))

    def processnewKinectIR(self,data):
        if self.isrecord:
            self.save_scheduler.update(data,"IRimage")
        img = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        qimg = qimage2ndarray.array2qimage(img)
        self.irLabel.setPixmap(QtGui.QPixmap(qimg))

    def button_record_clicked(self):
        if not self.isrecord:
            id = self.patientIDEdit.text()
            rectime = time.strftime('%Y-%m-%d_%H_%M_%S')
            newsavename = 'data/['+id+']'+rectime
            print(newsavename)
            self.save_thread = KinectSave(newsavename,fps=20)
            self.save_thread.start()
            self.save_Cloud_thread = HPS3DSave(newsavename,self.depth_stream)
            self.save_Cloud_thread.start()
            self.save_scheduler=KinectScheduler(self.save_thread,self.save_Cloud_thread, fps=20)
            self.save_scheduler.start()
            
            self.recordButton.setText("停止录制")
            self.isrecord=True
        else:
            print("release record")
            self.save_scheduler.stop()
            self.save_thread.stop()
            self.save_Cloud_thread.stop()
            


            self.recordButton.setText("开始录制")
            self.isrecord = False


if __name__ == '__main__':
    app = QApplication(sys.argv)
    openni2.initialize()
    dev = openni2.Device.open_all()
    dev = dev[0]
    depth_stream = dev.create_depth_stream()
    recordgui = RecordWindow(depth_stream)
    recordgui.show()
    sys.exit(app.exec_())
