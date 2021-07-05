import time
import pcl.pcl_visualization
from PyQt5.QtCore import QTimer, QThread, pyqtSignal

class DrawThread(QThread):
    def __init__(self, visual):
        super(DrawThread, self).__init__()
        self.active = True
        self.visual = visual
    def run(self):
        global cloud
        global visual
        while self.active:
            if cloud is not None:
                self.visual.ShowMonochromeCloud(cloud)
            time.sleep(0.1)