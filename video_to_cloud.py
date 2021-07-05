import time
from openni import openni2
import cv2
import numpy as np
import pcl
import pcl.pcl_visualization
import os
file = 'data\[001]2021-06-30_17_13_48'
visual = pcl.pcl_visualization.CloudViewing()

openni2.initialize()
dev = openni2.Device.open_all()
dev = dev[0]
depth_stream = dev.create_depth_stream()

length = len(list(os.listdir(file)))
for i in range(length):
    img_path = os.path.join(file , str(i)+'.png')
    print(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_ANYCOLOR)
    points = []
    for y in range(480):
            for x in range(640):
                c_x,c_y,c_z = openni2.convert_depth_to_world(depth_stream,x, y, img[y,x])
                if(c_z>=60000):
                    continue
                else:
                    point = [c_x/1000,c_y/1000,c_z/1000]
                    points.append(point)
    points = np.array(points).astype(np.float32)
    cloud = pcl.PointCloud(points)
    visual.ShowMonochromeCloud(cloud)
    time.sleep(1)



cap.release()
cv2.destroyAllWindows()