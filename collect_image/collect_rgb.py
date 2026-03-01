#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
海康威视单相机图像采集示例
--------------------------------

功能：
- 枚举当前连接的海康工业相机
- 选择一台相机（按索引）并打开
- 按固定时间间隔采集图像，持续指定时长
- 将图像保存为 JPG 文件到本地目录

注意：
- 依赖海康 MVS 的 Python SDK：MvCameraControl_class / MvErrorDefine_const / CameraParams_header
- 路径配置与原工程保持一致（../BasicDemo + MVCAM_COMMON_RUNENV 下的 MvImport）
- 构建了include包来优化代码结构
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

CAMERA_INDEX = 0        # 使用的相机索引（0 表示第一台）
SAVE_ROOT = "data"      # 图像保存根目录
CAPTURE_INTERVAL = 0.5  # 采集间隔（秒）
DURATION_SECONDS = 30   # 总采集时长（秒）

# ===========================
# 配置 MVS Python 接口搜索路径
# ===========================

#导入工程
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
    """
    简单的 memset，用于清零 C 结构体。
    dest: C 指针
    char: 填充值（int）
    size: 字节数
    """
    try:
        from ctypes import memmove
        memmove(dest, (c_ubyte * size)(*([char] * size)), size)
    except Exception:
        # 退化实现，逐字节写
        for i in range(size):
            dest[i] = char


def enum_cameras():
    """
    枚举当前可用相机。

    返回：
        device_list (MV_CC_DEVICE_INFO_LIST) 成功
        None 失败
    """
    if not CAMERA_AVAILABLE:
        print("未找到海康 MVS Python 库，请确认已安装并配置环境变量。")
        return None

    device_list = MV_CC_DEVICE_INFO_LIST()
    memset_py(byref(device_list), 0, sizeof(device_list))

    # 枚举 GigE + USB 设备
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, device_list)
    if ret != MV_OK:
        print(f"枚举设备失败，错误码: 0x{ret:x}")
        return None

    if device_list.nDeviceNum == 0:
        print("未发现任何相机。")
        return None

    print(f"发现 {device_list.nDeviceNum} 台相机：")
    for i in range(device_list.nDeviceNum):
        dev_info = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents

        # 这里只演示 GigE 情况，如有 USB 需要加判断
        model_name = "".join(
            chr(c) for c in dev_info.SpecialInfo.stGigEInfo.chModelName if c != 0
        )
        serial = "".join(
            chr(c) for c in dev_info.SpecialInfo.stGigEInfo.chSerialNumber if c != 0
        )
        print(f"  [{i}] Model: {model_name}, SN: {serial}")

    return device_list

def open_camera(device_list, index=0):
    """
    使用指定索引打开一台相机。

    参数：
        device_list : 枚举得到的 MV_CC_DEVICE_INFO_LIST
        index       : 相机索引

    返回：
        cam (MvCamera) 成功
        None 失败
    """
    if index < 0 or index >= device_list.nDeviceNum:
        print(f"相机索引 {index} 越界，可用范围: 0 ~ {device_list.nDeviceNum - 1}")
        return None

    cam = MvCamera()
    dev_info = cast(device_list.pDeviceInfo[index], POINTER(MV_CC_DEVICE_INFO)).contents

    # 创建句柄
    ret = cam.MV_CC_CreateHandle(dev_info)
    if ret != MV_OK:
        print(f"创建句柄失败，错误码: 0x{ret:x}")
        return None

# 打开设备
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != MV_OK:
        print(f"打开设备失败，错误码: 0x{ret:x}")
        cam.MV_CC_DestroyHandle()
        return None

    # =========== 【修改开始】 ===========
    # 尝试设置 BayerBG8 (这是海康大多数彩色相机的默认格式)
    # 使用字符串设置比十六进制更安全，SDK 会自动匹配
    print("正在尝试设置彩色格式...")
    ret = cam.MV_CC_SetEnumValueByString("PixelFormat", "BayerBG8")
    
    if ret == MV_OK:
        print("成功设置格式为: BayerBG8")
    else:
        # 如果 BayerBG8 也不行，尝试 BayerRG8 作为备选
        print(f"BayerBG8 设置失败 (0x{ret:x})，尝试 BayerRG8...")
        ret = cam.MV_CC_SetEnumValueByString("PixelFormat", "BayerRG8")
        if ret == MV_OK:
            print("成功设置格式为: BayerRG8")
        else:
            print(f"BayerRG8 也失败 (0x{ret:x})，尝试 BayerGR8...")
            ret = cam.MV_CC_SetEnumValueByString("PixelFormat", "BayerGR8")
            if ret != MV_OK:
                 print("警告：无法设置彩色格式，相机可能只支持黑白或特殊格式！")
    # =========== 【修改结束】 ===========

    # 设置为连续采集模式，关闭触发
    cam.MV_CC_SetEnumValueByString("AcquisitionMode", "Continuous")
    cam.MV_CC_SetEnumValueByString("TriggerMode", "Off")

    print(f"相机 {index} 已打开。")
    return cam


def start_grabbing(cam):
    """启动取流。"""
    ret = cam.MV_CC_StartGrabbing()
    if ret != MV_OK:
        print(f"开始取流失败，错误码: 0x{ret:x}")
        return False
    print("开始取流。")
    return True


def stop_and_close(cam):
    """停止取流并关闭设备，异常时忽略错误。"""
    try:
        cam.MV_CC_StopGrabbing()
    except Exception:
        pass
    try:
        cam.MV_CC_CloseDevice()
    except Exception:
        pass
    try:
        cam.MV_CC_DestroyHandle()
    except Exception:
        pass
    print("相机已关闭。")


def grab_one_rgb_image(cam):
    st_out_frame = MV_FRAME_OUT()
    memset_py(byref(st_out_frame), 0, sizeof(st_out_frame))

    ret = cam.MV_CC_GetImageBuffer(st_out_frame, 1000)
    if ret != MV_OK:
        print(f"获取图像失败，错误码: 0x{ret:x}")
        return None, False

    try:
        frame_len = st_out_frame.stFrameInfo.nFrameLen
        width = st_out_frame.stFrameInfo.nWidth
        height = st_out_frame.stFrameInfo.nHeight

        if width <= 0 or height <= 0:
            return None, False

        buf_addr = cast(st_out_frame.pBufAddr, POINTER(c_ubyte * frame_len))
        raw = bytes(buf_addr.contents)
        
        # 将原始数据转为 numpy 数组
        img = np.frombuffer(raw, dtype=np.uint8).reshape(height, width)

        # 【核心修改】进行 Bayer 解码
        # 海康相机通常默认为 BayerRG 顺序，所以使用 cv2.COLOR_BayerRG2BGR
        # 如果发现颜色不对（比如人脸发蓝），请尝试 BayerGB2BGR 或 BayerGR2BGR
        img = cv2.cvtColor(img, cv2.COLOR_BayerRG2BGR)

        return img, True
    finally:
        cam.MV_CC_FreeImageBuffer(st_out_frame)


def prepare_save_dir(root="data"):
    """
    创建带时间戳的保存目录。

    目录结构：
        root/single_cam_YYYYMMDD_HHMMSS/images/

    返回：
        img_dir (Path) 图像实际保存目录
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(root) / f"single_cam_{timestamp}"
    img_dir = save_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"图像将保存到: {img_dir}")
    return img_dir


def collect_images(duration_sec, interval_sec, camera_index=0, root="data_rgb"):
    """
    采集指定时长的图像。

    参数：
        duration_sec : 总采集时长（秒）
        interval_sec : 每两帧之间的时间间隔（秒）
        camera_index : 使用的相机索引（默认 0）
        root         : 根保存路径
    """
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

    print("开始采集... 按 Ctrl+C 可中途终止。")
    try:
        while True:
            now = time.time()
            if now - start_time >= duration_sec:
                break

            img, ok = grab_one_rgb_image(cam)
            if ok and img is not None:
                filename = img_dir / f"img_{frame_id:06d}.jpg"
                # 使用 OpenCV 保存为 JPEG
                cv2.imwrite(str(filename), img)
                print(f"[{frame_id}] 保存: {filename}")
                frame_id += 1
            else:
                print("本次抓图失败，跳过。")

            # 控制采集间隔
            time.sleep(interval_sec)

    except KeyboardInterrupt:
        print("用户中断采集。")
    finally:
        stop_and_close(cam)
        print(f"采集结束，共保存 {frame_id} 张图像。")


# ===========================
# 脚本入口
# ===========================

if __name__ == "__main__":
    # 简单示例：采集 DURATION_SECONDS 秒，间隔 CAPTURE_INTERVAL
    collect_images(
        duration_sec=DURATION_SECONDS,
        interval_sec=CAPTURE_INTERVAL,
        camera_index=CAMERA_INDEX,
        root=SAVE_ROOT,
    )
