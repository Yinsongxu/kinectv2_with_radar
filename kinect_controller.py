from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
import cv2
import sys,time,os,queue
import numpy as np
import copy


SKELETON_COLORS = [[0, 0, 255],  # red
                   [255, 0, 0],  # blue
                   [0, 255, 0],  # green
                   [255, 255, 255],  # white
                   [255, 0, 255],  # purple
                   [0, 255, 255],  # yellow
                   [0, 128, 128]]  # brown


class KinectSave(QThread):
    def __init__(self, name, fps=30, rgbdim=(1920, 1080), depthdim=(512, 424), fourcc="XVID"):
        super().__init__()
        self.active = True
        self.fps = fps
        self.RGBname = name + "-RGB.avi"
        self.Depthname = name + "-Depth.avi"
        self.IRname = name + "-IR.avi"
        self.Skname = name + "-skl.txt"
        self.rgbdim = rgbdim
        self.depthdim = depthdim
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self.RGBwriter = cv2.VideoWriter(self.RGBname, self.fourcc, self.fps, self.rgbdim)
        self.Depthwriter = cv2.VideoWriter(self.Depthname, self.fourcc, self.fps, self.depthdim)
        self.IRwriter = cv2.VideoWriter(self.IRname, self.fourcc, self.fps, self.depthdim, 0)
        self.Skfile = open(self.Skname, "a")
        self.framenum = 0
        self.RGBqueue = queue.Queue()
        self.Depthqueue = queue.Queue()
        self.IRqueue = queue.Queue()
        self.Skqueue = queue.Queue()

    def run(self):
        # time_prev = time.time()
        while self.active:
            while not self.RGBqueue.empty():
                RGBimage = self.RGBqueue.get()
                
                self.RGBwriter.write(RGBimage)

            while not self.Depthqueue.empty():
                Depthimage = self.Depthqueue.get()
                self.Depthwriter.write(Depthimage)

            while not self.IRqueue.empty():
                IRimage = self.IRqueue.get()
                self.IRwriter.write(IRimage)
            while not self.Skqueue.empty():
                Skdata = self.Skqueue.get()
                Sknbody = len(Skdata)
                Skstring = str(Sknbody) + "\n"
                for body in Skdata:
                    Skstring = Skstring + " ".join([str(entry) for entry in body[0]]) + "\n"
                    Skstring = Skstring + "25\n" + "\n".join(
                        [" ".join([str(entry) for entry in body[1][i]]) for i in range(25)]) + "\n"
                self.Skfile.write(Skstring)
                self.framenum = self.framenum + 1
            time.sleep(0.5)

        # postprocessing after stop
        while not self.RGBqueue.empty():
            RGBimage = self.RGBqueue.get()
            self.RGBwriter.write(RGBimage)
        print("1")

        while not self.Depthqueue.empty():
            Depthimage = self.Depthqueue.get()
            self.Depthwriter.write(Depthimage)
        print("2")
        while not self.IRqueue.empty():
            IRimage = self.IRqueue.get()
            self.IRwriter.write(IRimage)
        print("3")

        while not self.Skqueue.empty():
            Skdata = self.Skqueue.get()
            Sknbody = len(Skdata)
            Skstring = str(Sknbody) + "\n"
            for body in Skdata:
                Skstring = Skstring + " ".join([str(entry) for entry in body[0]]) + "\n"
                Skstring = Skstring + "25\n" + "\n".join(
                    [" ".join([str(entry) for entry in body[1][i]]) for i in range(25)]) + "\n"
            self.Skfile.write(Skstring)
            self.framenum = self.framenum + 1

        self.RGBwriter.release()
        self.Depthwriter.release()
        self.IRwriter.release()
        self.Skfile.close()
        with open(self.Skname, 'r') as original:
            Skdata_old = original.read()
        with open(self.Skname, 'w') as modified:
            modified.write(str(self.framenum) + "\n" + Skdata_old)

    def save(self, data, dtype):
        if self.active:
            if dtype == "RGBimage":
                self.RGBqueue.put(data)
            elif dtype == "Depthimage":
                self.Depthqueue.put(data)
            elif dtype == "IRimage":
                self.IRqueue.put(data)
            elif dtype == "Skdata":
                self.Skqueue.put(data)

    def stop(self):
        self.active = False


class KinectRefresh(QThread):
    newKinectRGB = pyqtSignal(object)
    newKinectDepth = pyqtSignal(object)
    newKinectIR = pyqtSignal(object)
    newKinectSk = pyqtSignal(object)
    newKinectRGBSk = pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.active = True
        self.kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color \
                                                      | PyKinectV2.FrameSourceTypes_Depth \
                                                      | PyKinectV2.FrameSourceTypes_Body \
                                                      | PyKinectV2.FrameSourceTypes_Infrared)

    def get_color_frame(self):
        if self.kinect.has_new_color_frame():
            colourframe = self.kinect.get_last_color_frame()
            colourframe = colourframe.astype(np.uint8)
            colourframe = np.reshape(colourframe, (1080, 1920, 4))
            framefullcolour = colourframe[:, :, :3]  # 1080, 1920, 3
            return framefullcolour

    def get_depth_frame(self):

        if self.kinect.has_new_depth_frame():
            depthframe = self.kinect.get_last_depth_frame()
            depth_frame = depthframe.reshape((424, 512,1)).astype(np.uint16)
            R = (depth_frame//256).astype(np.uint8)
            G = (depth_frame%256).astype(np.uint8)
            B = np.zeros_like(R).astype(np.uint8)
            depthframe = np.concatenate([B,G,R], 2)
            #depthframe = np.uint8(depth_frame / 8000 * 255)  # 424,512
            return depthframe

    def get_infrared_frame(self):
        if self.kinect.has_new_infrared_frame():
            infrared_frame = self.kinect.get_last_infrared_frame()
            infrared_frame = np.uint8(infrared_frame.clip(1, 4000) / 16.)
            infrared_frame = np.reshape(infrared_frame, (424, 512))  # 424,512
            return infrared_frame

    def get_body_frame(self):
        img = self.get_color_frame()
        body_frame = copy.deepcopy(img)
        if self.kinect.has_new_body_frame():
            bodies = self.kinect.get_last_body_frame()
            if bodies is not None:
                info = []
                for i in range(0, self.kinect.max_body_count):
                    body = bodies.bodies[i]
                    if not body.is_tracked:
                        continue
                    joints = body.joints
                    joint_orientations = body.joint_orientations
                    # convert joint coordinates to color space

                    joint_rgb_xy = self.kinect.body_joints_to_color_space(joints)
                    joint_depth_xy = self.kinect.body_joints_to_depth_space(joints)
                    '''
                    body_info_key =[
                                'bodyID', 'clipedEdges', 'handLeftConfidence',
                                'handLeftState', 'handRightConfidence', 'handRightState',
                                'isResticted', 'leanX', 'leanY', 'trackingState'
                            ]
                    '''

                    body_info = [body.tracking_id, body.clipped_edges, body.hand_left_confidence,
                                 body.hand_left_state, body.hand_right_confidence, body.hand_right_state,
                                 int(body.is_restricted), body.lean.x, body.lean.y, body.lean_tracking_state]

                    '''
                    joint_info_key = [
                                        'x', 'y', 'z', 
                                        'depthX', 'depthY', 'colorX', 'colorY',
                                        'orientationW', 'orientationX', 'orientationY','orientationZ',
                                        'trackingState'
                                    ]
                    '''
                    joint_info_all = []
                    for iq in range(25):
                        joint_info = [joints[iq].Position.x, joints[iq].Position.y, joints[iq].Position.z,
                                      joint_depth_xy[iq].x, joint_depth_xy[iq].y, joint_rgb_xy[iq].x,
                                      joint_rgb_xy[iq].y,
                                      joint_orientations[iq].Orientation.w, joint_orientations[iq].Orientation.x,
                                      joint_orientations[iq].Orientation.y, joint_orientations[iq].Orientation.z,
                                      joints[iq].TrackingState]
                        joint_info_all.append(joint_info)
                    self.draw_body(body_frame, joints, joint_rgb_xy, SKELETON_COLORS[i])
                    info.append([body_info, joint_info_all])
                    # print(points)
                return body_frame, img, info
            else:
                return body_frame, img, None
        else:
            return body_frame, img, None

    def draw_body_bone(self, img, joints, jointPoints, color, joint0, joint1):
        joint0State = joints[joint0].TrackingState
        joint1State = joints[joint1].TrackingState

        # both joints are not tracked
        if (joint0State == PyKinectV2.TrackingState_NotTracked) or (joint1State == PyKinectV2.TrackingState_NotTracked):
            return

        # both joints are not *really* tracked
        if (joint0State == PyKinectV2.TrackingState_Inferred) and (joint1State == PyKinectV2.TrackingState_Inferred):
            return

        # ok, at least one is good

        try:
            start = (int(jointPoints[joint0].x), int(jointPoints[joint0].y))
            end = (int(jointPoints[joint1].x), int(jointPoints[joint1].y))
            cv2.line(img, start, end, color, 4)

        except:  # need to catch it due to possible invalid positions (with inf)
            pass

    def draw_body(self, img, joints, jointPoints, color):
        # Torso
        self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_Head, PyKinectV2.JointType_Neck)
        self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_Neck,
                            PyKinectV2.JointType_SpineShoulder)
        self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder,
                            PyKinectV2.JointType_SpineMid)
        self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_SpineMid,
                            PyKinectV2.JointType_SpineBase)
        self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder,
                            PyKinectV2.JointType_ShoulderRight)
        self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder,
                            PyKinectV2.JointType_ShoulderLeft)
        self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_SpineBase,
                            PyKinectV2.JointType_HipRight)
        self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_SpineBase,
                            PyKinectV2.JointType_HipLeft)

        # Right Arm
        self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_ShoulderRight,
                            PyKinectV2.JointType_ElbowRight)
        self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_ElbowRight,
                            PyKinectV2.JointType_WristRight)
        self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_WristRight,
                            PyKinectV2.JointType_HandRight)
        self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_HandRight,
                            PyKinectV2.JointType_HandTipRight)
        self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_WristRight,
                            PyKinectV2.JointType_ThumbRight)

        # Left Arm
        self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_ShoulderLeft,
                            PyKinectV2.JointType_ElbowLeft)
        self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_ElbowLeft,
                            PyKinectV2.JointType_WristLeft)
        self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_WristLeft,
                            PyKinectV2.JointType_HandLeft)
        self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_HandLeft,
                            PyKinectV2.JointType_HandTipLeft)
        self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_WristLeft,
                            PyKinectV2.JointType_ThumbLeft)

        # Right Leg
        self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_HipRight,
                            PyKinectV2.JointType_KneeRight)
        self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_KneeRight,
                            PyKinectV2.JointType_AnkleRight)
        self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_AnkleRight,
                            PyKinectV2.JointType_FootRight)

        # Left Leg
        self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_HipLeft,
                            PyKinectV2.JointType_KneeLeft)
        self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_KneeLeft,
                            PyKinectV2.JointType_AnkleLeft)
        self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_AnkleLeft,
                            PyKinectV2.JointType_FootLeft)

    def run(self):
        while self.active:
            RGBSkimage, RGBimage, Skdata = self.get_body_frame()
            Depthimage = self.get_depth_frame()
            IRimage = self.get_infrared_frame()

            if RGBSkimage is not None:
                self.newKinectRGBSk.emit(RGBSkimage)
            if Skdata is not None:
                self.newKinectSk.emit(Skdata)
            if RGBimage is not None:
                self.newKinectRGB.emit(RGBimage)
            if IRimage is not None:
                self.newKinectIR.emit(IRimage)
            if Depthimage is not None:
                self.newKinectDepth.emit(Depthimage)
            time.sleep(0.01)

    def stop(self):
        self.active = False


class KinectScheduler(QThread):
    def __init__(self, save_thread, cloud_save, fps=20):
        super(KinectScheduler, self).__init__()
        self.inteval = 1 / fps
        self.timestart = None
        self.RGBimage = None
        self.Depthimage = None
        self.IRimage = None
        self.SkData = None
        self.Cloud = None
        self.active = True
        self.frameno = 0
        self.save_thread = save_thread
        self.cloud_save = cloud_save

    def run(self):
        self.timestart = time.time()
        while self.active:
            if time.time() - self.timestart >= self.frameno * self.inteval:
                if self.RGBimage is None or self.IRimage is None or self.Depthimage is None or self.SkData is None:
                    time.sleep(0.01)
                    #print("NoneSLEEP1")
                    continue
                #print("OK1")
                self.frameno = self.frameno + 1
                self.save_thread.save(self.RGBimage, "RGBimage")
                self.save_thread.save(self.IRimage, "IRimage")
                self.save_thread.save(self.Depthimage, "Depthimage")
                self.save_thread.save(self.SkData, "Skdata")
                self.cloud_save.save(self.Cloud)
            time.sleep(0.01)

    def update(self, data, type):
        if type == "RGBimage":
            self.RGBimage = data
        elif type == "IRimage":
            self.IRimage = data
        elif type == "Depthimage":
            self.Depthimage = data
        elif type == "Skdata":
            self.SkData = data
        elif type == "Cloud":
            self.Cloud = data

    def stop(self):
        self.active = False
