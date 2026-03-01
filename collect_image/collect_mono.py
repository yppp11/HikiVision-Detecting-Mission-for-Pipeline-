#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
海康威视单相机图像采集示例 - 纯灰度版本 (Mono8)
--------------------------------------------
说明：
- 强制设置相机为 Mono8 格式。
- 图像保存为单通道灰度 JPG（或为了兼容性转为3通道灰度）。
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from ctypes import *

# ===========================
# 用户可修改的基本参数
# ===========================

CAMERA_INDEX = 0        # 使用的相机索引
SAVE_ROOT = "data_mono" # 修改目录名区分彩色版
CAPTURE_INTERVAL = 0.5  # 采集间隔
DURATION_SECONDS = 30   # 采集时长

# ===========================
# 配置路径
# ===========================
sys.path.append("../BasicDemo")
sys.path.append(os.getenv("MVCAM_COMMON_RUNENV", "") + "/Samples/Python/MvImport")

try:
    from include.MvCameraControl_class import *
    from include.MvErrorDefine_const import *
    from include.CameraParams_header import *
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False

# ===========================
# 辅助函数
# ===========================

def memset_py(dest, char, size):
    try:
        from ctypes import memmove
        memmove(dest, (c_ubyte * size)(*([char] * size)), size)
    except Exception:
        for i in range(size):
            dest[i] = char

def enum_cameras():
    if not CAMERA_AVAILABLE:
        print("未找到 SDK。")
        return None

    device_list = MV_CC_DEVICE_INFO_LIST()
    memset_py(byref(device_list), 0, sizeof(device_list))
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, device_list)
    
    if ret != MV_OK or device_list.nDeviceNum == 0:
        print("未发现相机。")
        return None

    print(f"发现 {device_list.nDeviceNum} 台相机")
    return device_list

def open_camera(device_list, index=0):
    if index >= device_list.nDeviceNum:
        return None

    cam = MvCamera()
    dev_info = cast(device_list.pDeviceInfo[index], POINTER(MV_CC_DEVICE_INFO)).contents

    ret = cam.MV_CC_CreateHandle(dev_info)
    if ret != MV_OK:
        return None

    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != MV_OK:
        cam.MV_CC_DestroyHandle()
        return None

    # 【关键修改】强制设置为 Mono8 (黑白)
    # 这样可以减少带宽占用，且无需后续 Bayer 转换
    ret = cam.MV_CC_SetEnumValueByString("PixelFormat", "Mono8")
    if ret != MV_OK:
        print(f"警告: 设置 Mono8 失败 (0x{ret:x})，可能保持默认格式")

    cam.MV_CC_SetEnumValueByString("AcquisitionMode", "Continuous")
    cam.MV_CC_SetEnumValueByString("TriggerMode", "Off")

    print(f"相机 {index} 已打开 (Mono模式)。")
    return cam

def start_grabbing(cam):
    ret = cam.MV_CC_StartGrabbing()
    return ret == MV_OK

def stop_and_close(cam):
    try:
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
    except Exception:
        pass

def grab_one_mono_image(cam):
    """
    抓取一帧并返回灰度图像
    """
    st_out_frame = MV_FRAME_OUT()
    memset_py(byref(st_out_frame), 0, sizeof(st_out_frame))

    ret = cam.MV_CC_GetImageBuffer(st_out_frame, 1000)
    if ret != MV_OK:
        return None, False

    try:
        frame_len = st_out_frame.stFrameInfo.nFrameLen
        width = st_out_frame.stFrameInfo.nWidth
        height = st_out_frame.stFrameInfo.nHeight

        if width <= 0 or height <= 0:
            return None, False

        buf_addr = cast(st_out_frame.pBufAddr, POINTER(c_ubyte * frame_len))
        raw = bytes(buf_addr.contents)

        # 直接转为单通道灰度图
        img = np.frombuffer(raw, dtype=np.uint8).reshape(height, width)
        
        # 如果你需要保持和彩色版一样的 3通道数据结构 (虽然看起来是黑白的)，可以取消下面这行的注释
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return img, True
    finally:
        cam.MV_CC_FreeImageBuffer(st_out_frame)

def prepare_save_dir(root="data_mono"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(root) / f"single_cam_{timestamp}"
    img_dir = save_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    return img_dir

def collect_images(duration_sec, interval_sec, camera_index=0, root="data_mono"):
    device_list = enum_cameras()
    if device_list is None:
        return

    cam = open_camera(device_list, camera_index)
    if cam is None:
        return

    if not start_grabbing(cam):
        stop_and_close(cam)
        return

    img_dir = prepare_save_dir(root)
    start_time = time.time()
    frame_id = 0

    print("开始采集 (灰度)...")
    try:
        while True:
            now = time.time()
            if now - start_time >= duration_sec:
                break

            img, ok = grab_one_mono_image(cam)
            if ok and img is not None:
                filename = img_dir / f"img_{frame_id:06d}.jpg"
                cv2.imwrite(str(filename), img)
                print(f"[{frame_id}] 保存: {filename}")
                frame_id += 1
            
            time.sleep(interval_sec)

    except KeyboardInterrupt:
        print("用户中断。")
    finally:
        stop_and_close(cam)

if __name__ == "__main__":
    collect_images(DURATION_SECONDS, CAPTURE_INTERVAL, CAMERA_INDEX, SAVE_ROOT)