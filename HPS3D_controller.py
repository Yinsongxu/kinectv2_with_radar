import sys
import time
from HPS3D_IF import *
import numpy as np
import cv2
import time
import pcl.pcl_visualization
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from openni import openni2
import sys,time,os,queue
import pcl

class HPS3DSave(QThread):
    def __init__(self, name, depth_stream):
        super().__init__()
        self.active = True
        self.name = name
        if not os.path.exists(name):
            os.makedirs(name)
        self.framenum = 0
        self.depth_stream = depth_stream
        self.Cloudqueue = queue.Queue() 

    def frame_to_cloud(self, frame):
        points = []
        dframe_data = np.array(frame.get_buffer_as_triplet()).reshape([480, 640, 2])
        for y in range(frame.height):
            for x in range(frame.width):
                c_x,c_y,c_z = openni2.convert_depth_to_world(self.depth_stream,x+frame.cropOriginX, y+frame.cropOriginY, dframe_data[y,x,1]*255+dframe_data[y,x,0])
                if(c_z>=60000):
                    continue
                else:
                    point = [c_x/1000,c_y/1000,c_z/1000]
                    points.append(point)

        points = np.array(points).astype(np.float32)
        return points

    def run(self):
        # time_prev = time.time()
        while self.active:
            while not self.Cloudqueue.empty():
                frame = self.Cloudqueue.get()
                frame = frame.reshape(480, 640)
                cv2.imwrite(self.name+'/'+str(self.framenum)+'.png', frame)
                self.framenum = self.framenum + 1
            time.sleep(0.5)

        print('4')
        # postprocessing after stop
        while not self.Cloudqueue.empty():
            frame = self.Cloudqueue.get()
            frame = frame.reshape(480, 640)
            cv2.imwrite(self.name+'/'+str(self.framenum)+'.png', frame)
            self.framenum = self.framenum + 1
        print('5')

        '''
        self.Cloudfile.close()
        with open(self.name, 'r') as original:
            Cloud_old = original.read()
        with open(self.name, 'w') as modified:
            print('momomomo')
            modified.write(str(self.framenum) + "\n" + Cloud_old)
        '''

    def save(self, data):
        if self.active:
            self.Cloudqueue.put(data)

    def stop(self):
        print('cloud stop')
        self.active = False


class HPS3DRefresh(QThread):
    newCloud = pyqtSignal(object)
    def __init__(self, depth_stream, *args, **kwargs):
        super().__init__()
        self.active = True
        self.depth_stream = depth_stream
        self.depth_stream.start()

    def run(self):
        while self.active:
            frame = self.depth_stream.read_frame()
            dframe_data = np.array(frame.get_buffer_as_uint16()).reshape([480, 640,1])
            dframe_data[dframe_data>60000]=0
            if dframe_data is not None:
                self.newCloud.emit(dframe_data)
            time.sleep(0.01)

    def stop(self):
        self.active = False


