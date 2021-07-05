from ctypes import *

HPS_DLL = cdll.LoadLibrary(r'.\HPS3D_SDK.dll')

CDEBUGFUNC = CFUNCTYPE(None, c_char_p, c_void_p)  # 返回值：无 c_char_p：输出的Debug信息 c_void_p：用户自定义参数，回调上下文
COUTPUTEVENTFUNC = CFUNCTYPE(None, c_uint8, c_uint8,
                             c_void_p)  # 返回值：无 c_uint8：输出的设备ID c_uint8：输出的事件 c_void_p：用户自定义参数，回调上下文

# 全局常量
DEV_NUM = 0x04  # 最大连接设备数量
X_PIX_NUM = 640  # X轴像素个数
Y_PIX_NUM = 480  # Y轴像素个数
PIXEL_NUM = 307200  # 总像素个数

# 函数返回的结果
RET_OK = 0x01
RET_ERROR = 0x02
RET_BUSY = 0x03
RET_CONNECT_FAILED = 0x04  # 连接失败
RET_CREAT_PTHREAD_ERR = 0x05  # 线程创建失败
RET_WRITE_ERR = 0x06  # 数据发送失败
RET_READ_ERR = 0x07  # 读数据失败
RET_PACKET_HEAD_ERR = 0x08  # 数据包格式报头或报尾错误
RET_PACKET_ERR = 0x09  # 数据包解析错误
RET_BUFF_EMPTY = 0x0A  # 缓冲区为空或不可用
RET_VER_MISMATCH = 0x0B  # 固件版本与SDK版本不匹配
RET_INVALID_LICENSE = 0x0C  # 许可无效，需更新许可

# 运行模式
RUN_IDLE = 0x00  # 待机或停止测量
RUN_SINGLE_SHOT = 0x01  # 单次测量
RUN_CONTINUOUS = 0x02  # 连续测量

# 输出数据类型
OUTPUT_AMPLITUDE = 0x00  # 输出幅值数据
OUTPUT_DISTANCE_FULL = 0x01  # 输出完整深度数据
OUTPUT_DISTANCE_SIMPLE = 0x02  # 输出简单深度数据

# 平滑滤波器类型
SMOOTH_FILTER_DISABLE = 0x00  # 不使用平滑滤波器
SMOOTH_FILTER_AVERAGE = 0x01  # 滑动平均滤波器
SMOOTH_FILTER_GAUSS = 0x02  # 高斯平滑滤波器

# 点云图镜像模式
MIRROR_DISABLE = 0x00  # 不使用镜像
MIRROR_HORIZONTAL = 0x01  # 水平镜像
MIRROR_VERTICLA = 0x02  # 垂直镜像
MIRROR_ALL = 0x03  # 垂直和水平镜像

# 设备测量模式
DSITMODE_GENERAL = 0x01  # 普通模式
DSITMODE_SHORTDIST_HIGH_ACCURACY = 0x02  # 近距离高精度
DSITMODE_LONGDIST_HIGH_ACCURACY = 0x03  # 远距离高精度
DSITMODE_LONGDIST_LOW_ACCURACY = 0x04  # 远距离低精度

# 输出事件
EVENT_DEVDISCONNECT = 0x01  # 断开连接事件
EVENT_SIMPLEDISTANCERECVD = 0x02  # 简单深度数据接收事件
EVENT_FULLDISTANCERECVD = 0x03  # 完整深度数据接收事件
EVENT_AMPLITUDE = 0x04  # 振幅数据接收事件
EVENT_FULLPOINTCLOUDRECVD = 0x05  # 完整点云接收事件
EVENT_ROIPOINTCLOUDRECVD = 0x06  # ROI点云接收接收事件
EVENT_ROIFULLDISTANCERECVD = 0x07  # ROI完整深度接收事件
EVENT_ROISIMPLEDISTANCERECVD = 0x08  # ROI简单深度接收事件


def set_debug_callback(call_back, contex: c_void_p):
    """设置Debug回调函数

        参数:
            call_back: Debug回调函数，call_back设置为None时将停止该功能.

            contex: (c_void_p)用户自定义参数，回调上下文，不需要时则置为None.

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

        """
    ret_value = c_uint8(HPS_DLL.HPS3D_SetDebugCallBack(call_back, contex)).value
    dict = {'ret_value': ret_value}
    return dict


def connect_by_ethernet(server_ip: str, server_port: int):
    """设置以太网连接方式

        参数:
            server_ip: (string)设备的IP地址

            server_port: (int)设备的端口号

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value:　(int)返回信息，成功返回 RET_OK(1)

            device_id:  (int)设备ID号，连接失败时ID返回为99
        """
    device_id = c_uint8(99)
    ip = bytes(server_ip, "ascii")
    ret_value = c_uint8(HPS_DLL.HPS3D_ConnectByEthernet(ip, server_port, pointer(device_id))).value
    dict = {'ret_value': ret_value, 'device_id': device_id.value}
    return dict


def scan_device_list():
    """扫描设备的ID列表

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

            device_count: (int)已经连接设备的数量

            device_list: (list)存放已连接的设备ID号

            例如：当只有一台设备连接时
                device_count 返回值为1，device_list[0] = 当前已连接的设备ID
                device_list[1] = device_list[2] = device_list[3] = 99
        """
    array = c_uint8 * 4
    device_list = array()
    device_count = c_uint8(0)
    ret_value = c_uint8(HPS_DLL.HPS3D_ScanDeviceList(device_list, pointer(device_count))).value
    dict = {'ret_value': ret_value, 'device_count': device_count.value, 'device_list': list(device_list)}
    return dict


def disconnect(device_id: int):
    """断开设备,程序运行完成后需要调用此接口断开连接，释放资源

        参数:
            device_id: (int)设备ID,device_id必须与连接时返回的ID对应

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value:　(int)返回信息，成功返回 RET_OK(1)

        """
    ret_value = c_uint8(HPS_DLL.HPS3D_DisConnect(device_id)).value
    dict = {'ret_value': ret_value}
    return dict


def set_output_callback(call_back, device_id: int, contex: c_void_p):
    """设置输出事件回调函数

        参数:
            call_back: 输出事件回调函数，call_back设置为None时将停止该功能

            device_id: (int)设备ID,device_id必须与连接时返回的ID对应

            contex: (c_void_p)用户自定义参数，回调上下文，不需要时则置为None

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

        """
    ret_value = c_uint8(HPS_DLL.HPS3D_SetOutputCallBack(call_back, device_id, contex)).value
    dict = {'ret_value': ret_value}
    return dict


def get_average_distance_cb(device_id: int):
    """获取平均距离(回调函数中使用)

        注意:
            该接口只能在回调函数中收到 EVENT_SIMPLEDISTANCERECVD
            或者 EVENT_FULLDISTANCERECVD 或者 EVENT_FULLPOINTCLOUDRECVD 通知时调用

        参数:
            device_id: (int)设备ID,device_id必须与回调函数输出的ID对应

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

            aver_distance: (int)平均距离，单位毫米

        """
    aver_distance = c_uint16(0)
    ret_value = c_uint8(HPS_DLL.HPS3D_GetAverageDistance_CB(device_id, pointer(aver_distance))).value
    dict = {'ret_value': ret_value, 'aver_distance': aver_distance.value}
    return dict


def get_max_distance_cb(device_id: int):
    """获取最大距离(回调函数中使用)

        注意:
            该接口只能在回调函数中收到 EVENT_SIMPLEDISTANCERECVD
            或者 EVENT_FULLDISTANCERECVD 或者 EVENT_FULLPOINTCLOUDRECVD 通知时调用

        参数:
            device_id: (int)设备ID,device_id必须与回调函数输出的ID对应

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

            max_distance: (int)最大距离，单位毫米

        """
    max_distance = c_uint16(0)
    ret_value = c_uint8(HPS_DLL.HPS3D_GetMaxDistance_CB(device_id, pointer(max_distance))).value
    dict = {'ret_value': ret_value, 'max_distance': max_distance.value}
    return dict


def get_min_distance_cb(device_id: int):
    """获取最小距离(回调函数中使用)

        注意:
            该接口只能在回调函数中收到 EVENT_SIMPLEDISTANCERECVD
            或者 EVENT_FULLDISTANCERECVD 或者 EVENT_FULLPOINTCLOUDRECVD 通知时调用

        参数:
            device_id: (int)设备ID,device_id必须与回调函数输出的ID对应

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

            min_distance: (int)获取最小距离，单位毫米

        """
    min_distance = c_uint16(0)
    ret_value = c_uint8(HPS_DLL.HPS3D_GetMinDistance_CB(device_id, pointer(min_distance))).value
    dict = {'ret_value': ret_value, 'min_distance': min_distance.value}
    return dict


def get_all_average_amplitude_cb(device_id: int):
    """获取所有平均幅值(回调函数中使用)

        注意:
            该接口只能在回调函数中收到 EVENT_SIMPLEDISTANCERECVD
            或者 EVENT_FULLDISTANCERECVD 或者 EVENT_FULLPOINTCLOUDRECVD 通知时调用

        参数:
            device_id: (int)设备ID,device_id必须与回调函数输出的ID对应

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

            all_aver_amp: (int)所有平均幅值（包括0幅值）

        """
    all_aver_amp = c_uint16(0)
    ret_value = c_uint8(HPS_DLL.HPS3D_GetAllAverAmplitude_CB(device_id, pointer(all_aver_amp))).value
    dict = {'ret_value': ret_value, 'all_aver_amp': all_aver_amp.value}
    return dict


def get_valid_average_amplitude_cb(device_id: int):
    """获取有效的平均幅值(回调函数中使用)

        注意:
            该接口只能在回调函数中收到 EVENT_SIMPLEDISTANCERECVD
            或者 EVENT_FULLDISTANCERECVD 或者 EVENT_FULLPOINTCLOUDRECVD 通知时调用

        参数:
            device_id: (int)设备ID,device_id必须与回调函数输出的ID对应

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

            valid_aver_amp: (int)有效的平均幅值（不包括0幅值）

        """
    valid_aver_amp = c_uint16(0)
    ret_value = c_uint8(HPS_DLL.HPS3D_GetValidAverAmplitude_CB(device_id, pointer(valid_aver_amp))).value
    dict = {'ret_value': ret_value, 'valid_aver_amp': valid_aver_amp.value}
    return dict


def get_distance_data_cb(device_id: int):
    """获取距离数据(回调函数中使用)

        注意:
            该接口只能在回调函数中收到 EVENT_FULLDISTANCERECVD 通知时调用

        参数:
            device_id: (int)设备ID,device_id必须与回调函数输出的ID对应

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

            distance_data: (list)距离数据,单位毫米 (307200个空间 640 * 480)

        """
    array = c_uint16 * 307200
    distance_data = array()
    ret_value = c_uint8(HPS_DLL.HPS3D_GetDistanceData_CB(device_id, distance_data)).value
    dict = {'ret_value': ret_value, 'distance_data': list(distance_data)}
    return dict


def get_grayscale_data_cb(device_id: int):
    """获取灰度数据(回调函数中使用)

        注意:
            该接口只能在回调函数中收到 EVENT_AMPLITUDE 通知时调用

        参数:
            device_id: (int)设备ID,device_id必须与回调函数输出的ID对应

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

            grayscale_data: (list)灰度数据,单位灰度值(范围0~255) (307200个空间 640 * 480)

        """
    array = c_uint8 * 307200
    grayscale_data = array()
    ret_value = c_uint8(HPS_DLL.HPS3D_GetGrayscaleData_CB(device_id, grayscale_data)).value
    dict = {'ret_value': ret_value, 'grayscale_data': list(grayscale_data)}
    return dict


def get_amplitude_data_cb(device_id: int):
    """获取幅值数据(回调函数中使用)

        注意:
            该接口只能在回调函数中收到 EVENT_AMPLITUDE 通知时调用

        参数:
            device_id: (int)设备ID,device_id必须与回调函数输出的ID对应

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

            amplitude_data: (list)幅值数据 (307200个空间 640 * 480)

        """
    array = c_uint16 * 307200
    amplitude_data = array()
    ret_value = c_uint8(HPS_DLL.HPS3D_GetAmplitudeData_CB(device_id, amplitude_data)).value
    dict = {'ret_value': ret_value, 'amplitude_data': list(amplitude_data)}
    return dict


def get_min_amplitude_data_cb(device_id: int):
    """获取最小幅值数据(回调函数中使用)

        注意:
            该接口只能在回调函数中收到 EVENT_AMPLITUDE 通知时调用

        参数:
            device_id: (int)设备ID,device_id必须与回调函数输出的ID对应

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

            min_amplitude: (int)最小幅值数据

        """
    min_amplitude = c_uint16(0)
    ret_value = c_uint8(HPS_DLL.HPS3D_GetMinAmplitudeData_CB(device_id, pointer(min_amplitude))).value
    dict = {'ret_value': ret_value, 'min_amplitude': min_amplitude.value}
    return dict


def get_max_amplitude_data_cb(device_id: int):
    """获取最大幅值数据(回调函数中使用)

        注意:
            该接口只能在回调函数中收到 EVENT_AMPLITUDE 通知时调用

        参数:
            device_id: (int)设备ID,device_id必须与回调函数输出的ID对应

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

            max_amplitude: (int)最大幅值数据

        """
    max_amplitude = c_uint16(0)
    ret_value = c_uint8(HPS_DLL.HPS3D_GetMaxAmplitudeData_CB(device_id, pointer(max_amplitude))).value
    dict = {'ret_value': ret_value, 'max_amplitude': max_amplitude.value}
    return dict


def get_point_cloud_data_cb(device_id: int):
    """获取点云数据(回调函数中使用)

        注意:
            该接口只能在回调函数中收到 EVENT_FULLPOINTCLOUDRECVD 或者 EVENT_ROIPOINTCLOUDRECVD 通知时调用,
            无效点已经被自动过滤
        参数:
            device_id: (int)设备ID,device_id必须与回调函数输出的ID对应

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

            point_cloud_data: (list)点云数据,定义方式 point_data[point_count][3],其中数据个数由 point_count 返回值决定，
            [3]的值：0表示X轴值，1表示Y轴值，2表示Z轴值

            例如: point_cloud_data[0][0] 表示第一个点的X轴值
            point_cloud_data[1][1] 表示第二个点的Y轴值
            point_cloud_data[2][2] 表示第三个点的Z轴值

            point_cloud_count: (int)有效点的数量

        """
    SZ1 = c_double * 3
    SZ2 = SZ1 * 307200
    point_cloud_data = SZ2()
    point_cloud_count = c_uint32(0)
    ret_value = c_uint8(
        HPS_DLL.HPS3D_GetPointCloudData_CB(device_id, point_cloud_data, pointer(point_cloud_count))).value
    dict = {'ret_value': ret_value, 'point_cloud_count': point_cloud_count.value,
            'point_cloud_data': list(point_cloud_data)[0:point_cloud_count.value]}
    return dict


def get_out_roi_param_cb(device_id: int):
    """获取输出的ROI参数信息(回调函数中使用)

        注意:
            该接口只能在回调函数中收到 EVENT_ROIPOINTCLOUDRECVD 或者 EVENT_ROIFULLDISTANCERECVD
            或者 EVENT_ROISIMPLEDISTANCERECVD 通知时调用

        参数:
            device_id: (int)设备ID,device_id必须与回调函数输出的ID对应

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

            group_id: (int)ROI组ID

            roi_number: (int)ROI个数

            roi_id: (list)ROI的ID(8个空间),roi_number的值决定roi_id的有效数据个数，
                例如：roi_number = 1时
                roi_id[0] = ROI的ID。roi_id[1~7] 无效

        """
    group_id = c_uint8(0)
    roi_number = c_uint8(0)
    array = c_uint8 * 8
    roi_id = array()
    ret_value = c_uint8(
        HPS_DLL.HPS3D_GetOutRoiParam_CB(device_id, pointer(group_id), pointer(roi_number), roi_id)).value
    dict = {'ret_value': ret_value, 'group_id': group_id.value, 'roi_number': roi_number.value, 'roi_id': list(roi_id)}
    return dict


def get_out_roi_point_cloud_data_cb(device_id: int, roi_id: int):
    """获取输出ROI的点云数据(回调函数中使用)

        注意:
            该接口只能在回调函数中收到 EVENT_ROIPOINTCLOUDRECVD 通知时调用,无效点已经被自动过滤
            调用前先调用 get_out_roi_param_cb 接口获取对应的输入参数

        参数:
            device_id: (int)设备ID,device_id必须与回调函数输出的ID对应

            roi_id: (int)ROI的ID

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

            roi_point_data: (list)点云数据

            roi_point_count: (int)有效点数量
        """
    SZ1 = c_double * 3
    SZ2 = SZ1 * 307200
    roi_point_data = SZ2()
    roi_point_count = c_uint32(0)
    ret_value = c_uint8(
        HPS_DLL.HPS3D_GetOutRoiPointCloudData_CB(device_id, roi_id, roi_point_data, pointer(roi_point_count))).value
    dict = {'ret_value': ret_value, 'roi_point_count': roi_point_count.value,
            'roi_point_data': list(roi_point_data)[0:roi_point_count.value]}
    return dict


def get_out_roi_distance_data_cb(device_id: int, roi_id: int):
    """获取输出的ROI深度数据(回调函数中使用)

        注意:
            该接口只能在回调函数中收到 EVENT_ROIFULLDISTANCERECVD 通知时调用
            ,调用前先调用 get_out_roi_param_cb 接口获取对应的输入参数

        参数:
            device_id: (int)设备ID,device_id必须与回调函数输出的ID对应

            roi_id: (int)ROI的ID

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

            roi_distance_data: (list)ROI的深度数据，单位毫米(307200个空间)

            roi_pixel_number: (int)ROI的像素点数据个数

            roi_area: (list)ROI的范围：
                roi_area[0]：表示左上角点的X坐标

                roi_area[1]：表示左上角点的Y坐标

                roi_area[2]：表示右下角点的X坐标

                roi_area[3]：标志右下角点的Y坐标
        """
    array = c_uint16 * 307200
    roi_distance_data = array()
    array = c_uint16 * 4
    roi_area = array()
    ret_value = c_uint8(
        HPS_DLL.HPS3D_GetOutRoiDistanceData_CB(device_id, roi_id, roi_distance_data, roi_area)).value
    roi_pixel_number = (roi_area[2] - roi_area[0] + 1) * (roi_area[3] - roi_area[1] + 1)
    dict = {'ret_value': ret_value, 'roi_distance_data': list(roi_distance_data), 'roi_pixel_number': roi_pixel_number,
            'roi_area': list(roi_area)}
    return dict


def get_out_roi_average_distance_cb(device_id: int, roi_id: int):
    """获取输出的ROI平均距离(回调函数中使用)

        注意:
            该接口只能在回调函数中收到 EVENT_ROIFULLDISTANCERECVD 或者
            EVENT_ROISIMPLEDISTANCERECVD 或者 EVENT_ROIPOINTCLOUDRECVD 通知时调用通知时调用,
            调用前先调用 get_out_roi_param_cb 接口获取对应的输入参数

        参数:
            device_id: (int)设备ID,device_id必须与回调函数输出的ID对应

            roi_id: (int)ROI的ID

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

            aver_distance: (int)平均距离，单位毫米
        """
    aver_distance = c_uint16(0)
    ret_value = c_uint8(HPS_DLL.HPS3D_GetOutRoiAverageDistance_CB(device_id, roi_id, pointer(aver_distance))).value
    dict = {'ret_value': ret_value, 'aver_distance': aver_distance.value}
    return dict


def get_out_roi_max_distance_cb(device_id: int, roi_id: int):
    """获取输出的ROI最大距离(回调函数中使用)

        注意:
            该接口只能在回调函数中收到 EVENT_ROIFULLDISTANCERECVD 或者
            EVENT_ROISIMPLEDISTANCERECVD 或者 EVENT_ROIPOINTCLOUDRECVD 通知时调用通知时调用，
            调用前先调用 get_out_roi_param_cb 接口获取对应的输入参数

        参数:
            device_id: (int)设备ID,device_id必须与回调函数输出的ID对应

            roi_id: (int)ROI的ID

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

            max_distance: (int)最大距离，单位毫米
        """
    max_distance = c_uint16(0)
    ret_value = c_uint8(HPS_DLL.HPS3D_GetOutRoiMaxDistance_CB(device_id, roi_id, pointer(max_distance))).value
    dict = {'ret_value': ret_value, 'max_distance': max_distance.value}
    return dict


def get_out_roi_min_distance_cb(device_id: int, roi_id: int):
    """获取输出的ROI最小距离(回调函数中使用)

        注意:
            该接口只能在回调函数中收到 EVENT_ROIFULLDISTANCERECVD 或者
            EVENT_ROISIMPLEDISTANCERECVD 或者 EVENT_ROIPOINTCLOUDRECVD 通知时调用通知时调用，
            调用前先调用 get_out_roi_param_cb 接口获取对应的输入参数

        参数:
            device_id: (int)设备ID,device_id必须与回调函数输出的ID对应

            roi_id: (int)ROI的ID

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

            min_distance: (int)最小距离，单位毫米
        """
    min_distance = c_uint16(0)
    ret_value = c_uint8(HPS_DLL.HPS3D_GetOutRoiMinDistance_CB(device_id, roi_id, pointer(min_distance))).value
    dict = {'ret_value': ret_value, 'min_distance': min_distance.value}
    return dict


def get_out_roi_all_average_amplitude_cb(device_id: int, roi_id: int):
    """获取输出的ROI所有平均幅值(回调函数中使用)

        注意:
            该接口只能在回调函数中收到 EVENT_ROIFULLDISTANCERECVD 或者
            EVENT_ROISIMPLEDISTANCERECVD 或者 EVENT_ROIPOINTCLOUDRECVD 通知时调用通知时调用，
            调用前先调用 get_out_roi_param_cb 接口获取对应的输入参数

        参数:
            device_id: (int)设备ID,device_id必须与回调函数输出的ID对应

            roi_id: (int)ROI的ID

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

            all_aver_amp: (int)所有平均幅值（包括0幅值）
        """
    all_aver_amp = c_uint16(0)
    ret_value = c_uint8(HPS_DLL.HPS3D_GetOutRoiAllAverAmplitude_CB(device_id, roi_id, pointer(all_aver_amp))).value
    dict = {'ret_value': ret_value, 'all_aver_amp': all_aver_amp.value}
    return dict


def get_out_roi_valid_average_amplitude_cb(device_id: int, roi_id: int):
    """获取输出的ROI有效平均幅值(回调函数中使用)

        注意:
            该接口只能在回调函数中收到 EVENT_ROIFULLDISTANCERECVD 或者
            EVENT_ROISIMPLEDISTANCERECVD 或者 EVENT_ROIPOINTCLOUDRECVD 通知时调用通知时调用，
            调用前先调用 get_out_roi_param_cb 接口获取对应的输入参数

        参数:
            device_id: (int)设备ID,device_id必须与回调函数输出的ID对应

            roi_id: (int)ROI的ID

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

            valid_aver_amp: (int)有效平均幅值（不包括0幅值）
        """
    valid_aver_amp = c_uint16(0)
    ret_value = c_uint8(
        HPS_DLL.HPS3D_GetOutRoiValidAverAmplitude_CB(device_id, roi_id, pointer(valid_aver_amp))).value
    dict = {'ret_value': ret_value, 'valid_aver_amp': valid_aver_amp.value}
    return dict


def set_run_mode(device_id: int, run_mode: int):
    """设置运行模式

        注意:
            每次设置为 RUN_SINGLE_SHOT 将通过回调函数吐一帧数据

        参数:
            device_id: (int)设备ID,device_id必须与连接时返回的ID对应

            run_mode: (int)运行模式，参照 HPS3D_IF.py中的定义

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

        """
    ret_value = c_uint8(HPS_DLL.HPS3D_SetRunMode(device_id, run_mode)).value
    dict = {'ret_value': ret_value}
    return dict


def get_run_mode(device_id: int):
    """设置运行模式

        注意:
            每次设置为 RUN_SINGLE_SHOT 将通过回调函数吐一帧数据

        参数:
            device_id: (int)设备ID,device_id必须与连接时返回的ID对应

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

            run_mode: (int)运行模式，参照 HPS3D_IF.py中的定义
        """
    run_mode = c_uint8(0)
    ret_value = c_uint8(HPS_DLL.HPS3D_GetRunMode(device_id, pointer(run_mode))).value
    dict = {'ret_value': ret_value, 'run_mode': run_mode.value}
    return dict


def set_point_cloud_mode(device_id: int, point_cloud_enable: bool, mirror_mode: int):
    """设置点云模式配置

        注意:
            执行该命令前，需要将运行模式设置为 RUN_IDLE

        参数:
            device_id: (int)设备ID,device_id必须与连接时返回的ID对应

            point_cloud_enable: (bool)点云图使能

            mirror_mode: (int)点云图镜像模式,参照 HPS3D_IF.py中的定义

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)
        """
    ret_value = c_uint8(HPS_DLL.HPS3D_SetPointCloudMode(device_id, point_cloud_enable, mirror_mode)).value
    dict = {'ret_value': ret_value}
    return dict


def get_point_cloud_mode(device_id: int):
    """获取点云模式配置

        注意:
            执行该命令前，需要将运行模式设置为 RUN_IDLE

        参数:
            device_id: (int)设备ID,device_id必须与连接时返回的ID对应

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

            enable: (bool)点云图使能

            mode: (int)点云图镜像模式,参照 HPS3D_IF.py中的定义
        """
    point_cloud_enable = c_bool(False)
    mirror_mode = c_uint8(0)
    ret_value = c_uint8(
        HPS_DLL.HPS3D_GetPointCloudMode(device_id, pointer(point_cloud_enable), pointer(mirror_mode))).value
    dict = {'ret_value': ret_value, 'enable': point_cloud_enable.value, 'mode': mirror_mode.value}
    return dict


def set_measure_mode(device_id: int, measure_mode: int):
    """设置测量模式

        注意:
            执行该命令前，需要将运行模式设置为 RUN_IDLE

        参数:
            device_id: (int)设备ID,device_id必须与连接时返回的ID对应

            measure_mode: (int)测量模式,参照 HPS3D_IF.py中的定义

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)
        """
    ret_value = c_uint8(HPS_DLL.HPS3D_SetMeasureMode(device_id, measure_mode)).value
    dict = {'ret_value': ret_value}
    return dict


def get_measure_mode(device_id: int):
    """获取测量模式

        注意:
            执行该命令前，需要将运行模式设置为 RUN_IDLE

        参数:
            device_id: (int)设备ID,device_id必须与连接时返回的ID对应

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

            mode: (int)测量模式,参照 HPS3D_IF.py中的定义

        """
    measure_mode = c_uint8(0)
    ret_value = c_uint8(
        HPS_DLL.HPS3D_GetMeasureMode(device_id, pointer(measure_mode))).value
    dict = {'ret_value': ret_value, 'mode': measure_mode.value}
    return dict


def get_sdk_version():
    """获取SDK版本信息

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value：　(int)返回信息，成功返回 RET_OK(1)

            version_year:　(int)SDK版本信息：年

            version_month：　(int)SDK版本信息：月

            version_day：　(int)SDK版本信息：日

            version_major：　(int)SDK版本信息：大版本

            version_minor：　(int)SDK版本信息：小版本

            version_rev：　(int)SDK版本信息：修订

        """
    array = c_uint8 * 6
    version = array()
    ret_value = c_uint8(HPS_DLL.HPS3D_GetSDKVersion(version)).value
    dict = {'ret_value': ret_value, 'version_year': version[0], 'version_month': version[1], 'version_day': version[2],
            'version_major': version[3], 'version_minor': version[4], 'version_rev': version[5]}
    return dict


def get_device_version(device_id: int):
    """获取设备版本信息

        注意:
            执行该命令前，需要将运行模式设置为 RUN_IDLE

        参数:
            device_id: (int)设备ID,device_id必须与连接时返回的ID对应

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value：　(int)返回信息，成功返回 RET_OK(1)

            version_year：　(int)SDK版本信息：年

            version_month：　(int)SDK版本信息：月

            version_day：　(int)SDK版本信息：日

            version_major：　(int)SDK版本信息：大版本

            version_minor：　(int)SDK版本信息：小版本

            version_rev：　(int)SDK版本信息：修订

        """
    array = c_uint8 * 6
    version = array()
    ret_value = c_uint8(HPS_DLL.HPS3D_GetDeviceVersion(device_id, version)).value
    dict = {'ret_value': ret_value, 'version_year': version[0], 'version_month': version[1], 'version_day': version[2],
            'version_major': version[3], 'version_minor': version[4], 'version_rev': version[5]}
    return dict


def get_device_sn(device_id: int):
    """获取设备序列号

        注意:
            执行该命令前，需要将运行模式设置为 RUN_IDLE

        参数:
            device_id: (int)设备ID,device_id必须与连接时返回的ID对应

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

            sn: (str)设备的SN

        """
    array = c_char * 64
    sn = array()
    ret_value = c_uint8(HPS_DLL.HPS3D_GetDeviceSN(device_id, sn)).value
    dict = {'ret_value': ret_value, 'sn': sn.value.decode().replace('-', '')}
    return dict


def get_output_data_type(device_id: int):
    """获取输出数据类型

        注意:
            执行该命令前，需要将运行模式设置为 RUN_IDLE

        参数:
            device_id: (int)设备ID,device_id必须与连接时返回的ID对应

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

            type: (int)输出数据类型,参照 HPS3D_IF.py中的定义

        """
    data_type = c_uint8(0)
    ret_value = c_uint8(HPS_DLL.HPS3D_GetOutputDataType(device_id, pointer(data_type))).value
    dict = {'ret_value': ret_value, 'type': data_type.value}
    return dict


def set_output_data_type(device_id: int, data_type: int):
    """设置输出数据类型

        注意:
            执行该命令前，需要将运行模式设置为 RUN_IDLE

        参数:
            device_id: (int)设备ID,device_id必须与连接时返回的ID对应

            data_type: (int)输出数据类型,参照 HPS3D_IF.py中的定义

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

        """
    ret_value = c_uint8(HPS_DLL.HPS3D_SetOutputDataType(device_id, data_type)).value
    dict = {'ret_value': ret_value}
    return dict


def set_integ_time(device_id: int, time_us: int):
    """设置积分时间

        注意:
            执行该命令前，需要将运行模式设置为 RUN_IDLE

        参数:
            device_id: (int)设备ID,device_id必须与连接时返回的ID对应

            time_us: (int)积分时间(单位us，范围50~1000)

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

        """
    ret_value = c_uint8(HPS_DLL.HPS3D_SetIntegTime(device_id, time_us)).value
    dict = {'ret_value': ret_value}
    return dict


def get_integ_time(device_id: int):
    """获取积分时间

        注意:
            执行该命令前，需要将运行模式设置为 RUN_IDLE

        参数:
            device_id: (int)设备ID,device_id必须与连接时返回的ID对应

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

            time_us: (int)积分时间(单位us，范围50~1000)

        """
    time_us = c_uint32(0)
    ret_value = c_uint8(HPS_DLL.HPS3D_GetIntegTime(device_id, pointer(time_us))).value
    dict = {'ret_value': ret_value, 'time_us': time_us.value}
    return dict


def profile_save_to_user(device_id: int):
    """保存配置到用户设置表

        注意:
            执行该命令前，需要将运行模式设置为 RUN_IDLE

        参数:
            device_id: (int)设备ID,device_id必须与连接时返回的ID对应

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

        """
    ret_value = c_uint8(HPS_DLL.HPS3D_ProfileSaveToUser(device_id)).value
    dict = {'ret_value': ret_value}
    return dict


def profile_clear_user(device_id: int):
    """清除用户设置表

        注意:
            执行该命令前，需要将运行模式设置为 RUN_IDLE

        参数:
            device_id: (int)设备ID,device_id必须与连接时返回的ID对应

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

        """
    ret_value = c_uint8(HPS_DLL.HPS3D_ProfileClearUser(device_id)).value
    dict = {'ret_value': ret_value}
    return dict


def profile_restore_factory(device_id: int):
    """还原出厂设置

        注意:
            执行该命令前，需要将运行模式设置为 RUN_IDLE

        参数:
            device_id: (int)设备ID,device_id必须与连接时返回的ID对应

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

        """
    ret_value = c_uint8(HPS_DLL.HPS3D_ProfileRestoreFactory(device_id)).value
    dict = {'ret_value': ret_value}
    return dict


def set_roi_group(device_id: int, group_id: int):
    """选择ROI组

        注意:
            执行该命令前，需要将运行模式设置为 RUN_IDLE

        参数:
            device_id: (int)设备ID,device_id必须与连接时返回的ID对应

            group_id: (int)ROI组ID，范围0~1

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

        """
    ret_value = c_uint8(HPS_DLL.HPS3D_SetROIGroup(device_id, group_id)).value
    dict = {'ret_value': ret_value}
    return dict


def get_roi_group(device_id: int):
    """获取当前的ROI组ID

        注意:
            执行该命令前，需要将运行模式设置为 RUN_IDLE

        参数:
            device_id: (int)设备ID,device_id必须与连接时返回的ID对应

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

            group_id: (int)ROI组ID，范围0~1

        """
    group_id = c_uint8(0)
    ret_value = c_uint8(HPS_DLL.HPS3D_SetROIGroup(device_id, pointer(group_id))).value
    dict = {'ret_value': ret_value, 'group_id': group_id.value}
    return dict


def set_roi_enable(device_id: int, roi_id: int, enable: bool):
    """选择ROI组

        注意:
            执行该命令前，需要将运行模式设置为 RUN_IDLE

        参数:
            device_id: (int)设备ID,device_id必须与连接时返回的ID对应

            roi_id: (int)ROI的ID

            enable: (bool)ROI使能

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

        """
    ret_value = c_uint8(HPS_DLL.HPS3D_SetROIEnable(device_id, roi_id, enable)).value
    dict = {'ret_value': ret_value}
    return dict


def get_roi_enable(device_id: int, roi_id: int):
    """获取当前的ROI组ID

        注意:
            执行该命令前，需要将运行模式设置为 RUN_IDLE

        参数:
            device_id: (int)设备ID,device_id必须与连接时返回的ID对应

            roi_id: (int)ROI的ID

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

            enable: (bool)ROI使能

        """
    enable = c_bool(False)
    ret_value = c_uint8(HPS_DLL.HPS3D_GetROIEnable(device_id, roi_id, pointer(enable))).value
    dict = {'ret_value': ret_value, 'enable': enable.value}
    return dict


def get_number_of_roi(device_id: int):
    """获取当前设备支持的ROI数量和阈值数量

        注意:
            执行该命令前，需要将运行模式设置为 RUN_IDLE

        参数:
            device_id: (int)设备ID,device_id必须与连接时返回的ID对应

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

            roi_number: (int)支持的ROI数量

            threshold_number: (int)支持的阈值数量

            roi_number_group: (int)支持的ROI组数量

        """
    roi_number = c_uint8(0)
    threshold_number = c_uint8(0)
    roi_number_group = c_uint8(0)
    ret_value = c_uint8(HPS_DLL.HPS3D_GetNumberOfROI(device_id, pointer(roi_number), pointer(threshold_number),
                                                     pointer(roi_number_group))).value
    dict = {'ret_value': ret_value, 'roi_number': roi_number.value, 'threshold_number': threshold_number.value,
            'roi_number_group': roi_number_group.value}
    return dict


def set_smooth_filter(device_id: int, filter_type: int, filter_times: int):
    """设置平滑滤波器

        注意:
            执行该命令前，需要将运行模式设置为 RUN_IDLE

        参数:
            device_id: (int)设备ID,device_id必须与连接时返回的ID对应

            filter_type: (int)平滑滤波器类型，参照 HPS3D_IF.py中的定义

            filter_times: (int)平滑滤波器滤波次数，范围0~10

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

        """
    ret_value = c_uint8(HPS_DLL.HPS3D_SetSmoothFilter(device_id, filter_type, filter_times)).value
    dict = {'ret_value': ret_value}
    return dict


def get_smooth_filter(device_id: int):
    """获取平滑滤波器设置

        注意:
            执行该命令前，需要将运行模式设置为 RUN_IDLE

        参数:
            device_id: (int)设备ID,device_id必须与连接时返回的ID对应

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

            type: (int)平滑滤波器类型，参照 HPS3D_IF.py中的定义

            times: (int)平滑滤波器滤波次数，范围0~10

        """
    filter_type = c_uint8(0)
    filter_times = c_uint32(0)
    ret_value = c_uint8(
        HPS_DLL.HPS3D_GetSmoothFilter(device_id, pointer(filter_type), pointer(filter_times))).value
    dict = {'ret_value': ret_value, 'type': filter_type.value, 'times': filter_times.value}
    return dict


def software_reset(device_id: int):
    """软件复位

        注意:
            执行该命令前，需要将运行模式设置为 RUN_IDLE

        参数:
            device_id: (int)设备ID,device_id必须与连接时返回的ID对应

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

        """
    ret_value = c_uint8(HPS_DLL.HPS3D_SoftwareReset(device_id)).value
    dict = {'ret_value': ret_value}
    return dict


def set_offset(device_id: int, offset_mm: int):
    """设置距离偏移

        注意:
            执行该命令前，需要将运行模式设置为 RUN_IDLE

        参数:
            device_id: (int)设备ID,device_id必须与连接时返回的ID对应

            offset_mm: (int)偏移距离，单位MM

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

        """
    ret_value = c_uint8(HPS_DLL.HPS3D_SetOffset(device_id, offset_mm)).value
    dict = {'ret_value': ret_value}
    return dict


def get_offset(device_id: int):
    """获取距离偏移

        注意:
            执行该命令前，需要将运行模式设置为 RUN_IDLE

        参数:
            device_id: (int)设备ID,device_id必须与连接时返回的ID对应

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

            offset_mm: (int)偏移距离，单位MM

        """
    offset_mm = c_int16(0)
    ret_value = c_uint8(HPS_DLL.HPS3D_GetOffset(device_id, pointer(offset_mm))).value
    dict = {'ret_value': ret_value, 'offset_mm': offset_mm.value}
    return dict


def set_simple_kalman_filter(device_id: int, kalman_enable: bool, kalman_k: float, kalman_number: int,
                             kalman_threshold: int):
    """设置卡尔曼滤波器

        注意:
            执行该命令前，需要将运行模式设置为 RUN_IDLE

        参数:
            device_id: (int)设备ID,device_id必须与连接时返回的ID对应

            kalman_enable: (bool)卡尔曼滤波器使能

            kalman_k: (float)卡尔曼滤波器比例系数

            kalman_number: (int)卡尔曼滤波器阈值检查帧数

            kalman_threshold: (int)卡尔曼滤波器噪声阈值

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

        """
    ret_value = c_uint8(HPS_DLL.HPS3D_SetSimpleKalmanFilter(device_id, kalman_enable, kalman_k, kalman_number,
                                                            kalman_threshold)).value
    dict = {'ret_value': ret_value}
    return dict


def get_simple_kalman_filter(device_id: int):
    """获取卡尔曼滤波器配置

        注意:
            执行该命令前，需要将运行模式设置为 RUN_IDLE

        参数:
            device_id: (int)设备ID,device_id必须与连接时返回的ID对应

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

            enable: (bool)卡尔曼滤波器使能

            k: (float)卡尔曼滤波器比例系数

            number: (int)卡尔曼滤波器阈值检查帧数

            threshold: (int)卡尔曼滤波器噪声阈值

        """
    kalman_enable = c_bool(False)
    kalman_k = c_double(0.0)
    kalman_number = c_uint32(0)
    kalman_threshold = c_uint32(0)
    ret_value = c_uint8(HPS_DLL.HPS3D_GetSimpleKalmanFilter(device_id, pointer(kalman_enable), pointer(kalman_k),
                                                            pointer(kalman_number),
                                                            pointer(kalman_threshold))).value
    dict = {'ret_value': ret_value, 'enable': kalman_enable.value, 'k': kalman_k.value,
            'number': kalman_number.value, 'threshold': kalman_threshold.value}
    return dict


def set_watchdog_enable(device_id: int, enable: bool):
    """设置看门狗使能

        注意:
            执行该命令前，需要将运行模式设置为 RUN_IDLE
            单步调试程序时需要将看门狗使能关闭，否则可能会导致设备断连
            正式运行程序时需要将看门狗使能打开
        参数:
            device_id: (int)设备ID,device_id必须与连接时返回的ID对应

            enable: (bool) 看门狗使能

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

        """
    ret_value = c_uint8(HPS_DLL.HPS3D_SetWatchDogEnable(device_id, enable)).value
    dict = {'ret_value': ret_value}
    return dict


def get_watchdog_enable(device_id: int):
    """获取看门狗使能

        注意:
            执行该命令前，需要将运行模式设置为 RUN_IDLE

        参数:
            device_id: (int)设备ID,device_id必须与连接时返回的ID对应

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

            enable: (bool)看门狗使能

        """
    enable = c_bool(False)
    ret_value = c_uint8(HPS_DLL.HPS3D_GetWatchDogEnable(device_id, pointer(enable))).value
    dict = {'ret_value': ret_value, 'enable': enable.value}
    return dict


def get_edge_filter(device_id: int):
    """获取边缘滤波器设置

        参数:
            device_id: (int)设备ID,device_id必须与连接时返回的ID对应

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

            enable: (bool)边缘滤波器使能

            args: (int)边缘滤波器参数
        """
    enable = c_bool(False)
    args = c_uint32(0)
    ret_value = c_uint8(HPS_DLL.HPS3D_GetEdgeFilter(device_id, pointer(enable), pointer(args))).value
    dict = {'ret_value': ret_value, 'enable': enable.value, 'args':args.value}
    return dict

def set_edge_filter(device_id: int, enable: bool, args: int):
    """设置边缘滤波器设置

        参数:
            device_id: (int)设备ID,device_id必须与连接时返回的ID对应

            enable: (bool)边缘滤波器使能

            args: (int)边缘滤波器参数

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

        """
    ret_value = c_uint8(HPS_DLL.HPS3D_SetEdgeFilter(device_id, enable, args)).value
    dict = {'ret_value': ret_value}
    return dict

def get_hdr_enable(device_id: int):
    """获取HDR模式使能

        参数:
            device_id: (int)设备ID,device_id必须与连接时返回的ID对应

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

            enable: (bool)HDR模式使能

        """
    enable = c_bool(False)
    ret_value = c_uint8(HPS_DLL.HPS3D_GetHDREnable(device_id, pointer(enable))).value
    dict = {'ret_value': ret_value, 'enable': enable.value}
    return dict

def set_hdr_enable(device_id: int, enable: bool):
    """设置HDR模式使能

        参数:
            device_id: (int)设备ID,device_id必须与连接时返回的ID对应

            enable: (bool)HDR模式使能

        返回值:
            该函数将返回一个字典，通过键值映射能够从字典中获取相应值

            ret_value: (int)返回信息，成功返回 RET_OK(1)

        """
    ret_value = c_uint8(HPS_DLL.HPS3D_SetHDREnable(device_id, enable)).value
    dict = {'ret_value': ret_value}
    return dict