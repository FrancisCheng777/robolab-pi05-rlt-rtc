#!/usr/bin/env python3
"""相机封装：Kinect v2（第三视角）+ RealSense D435i（腕部）。

Kinect v2 的关键事实：
  - 不是 UVC 设备，没有 /dev/video 节点，必须用 libfreenect2
  - 原始输出 1920x1080，画面需要顺时针旋转 90 度
  - 旋转后是 1080x1920（竖长），需居中裁成正方形再缩放

RealSense D435i：
  - 走 pyrealsense2，输出 BGR，需转 RGB
"""
from __future__ import annotations

import ctypes
import os
import sys
import time

import numpy as np

# libfreenect2 的位置（roboLab 自带）
FREENECT2_ROOT = os.path.expanduser("~/roboLab/build/third_party/libfreenect2")
FREENECT2_LIB = os.path.join(FREENECT2_ROOT, "lib")
if os.path.isdir(FREENECT2_LIB):
    os.environ["LD_LIBRARY_PATH"] = FREENECT2_LIB + ":" + os.environ.get("LD_LIBRARY_PATH", "")


# ---------------- Kinect v2 ----------------
def _preload_freenect2():
    """提前用 ctypes 载入 libfreenect2，让 pylibfreenect2 能找到它。"""
    path = os.path.join(FREENECT2_LIB, "libfreenect2.so.0.2")
    if os.path.exists(path):
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            return True
        except OSError:
            return False
    return False


class KinectV2:
    """Kinect v2 RGB 读取器。输出已旋转为正向的 HWC uint8 RGB。"""

    def __init__(self, serial: str | None = None, rotate_cw: bool = True, crop_square: bool = True):
        _preload_freenect2()
        from pylibfreenect2 import (
            Freenect2, SyncMultiFrameListener, Frame, FrameType,
            CpuPacketPipeline, Registration,
        )
        self._rt = (Freenect2, SyncMultiFrameListener, Frame, FrameType, Registration)
        self.rotate_cw = rotate_cw
        self.crop_square = crop_square

        self.fn = Freenect2()
        n = self.fn.enumerateDevices()
        if n == 0:
            raise RuntimeError("未发现 Kinect v2 设备")
        self.serial = serial or self.fn.getDefaultDeviceSerialNumber()
        self.device = self.fn.openDevice(self.serial, CpuPacketPipeline())
        if self.device is None:
            raise RuntimeError("打开 Kinect v2 失败: " + self.serial)

        # 只开彩色流；pylibfreenect2 的 start()/startStreams() 返回 None，不能当布尔判断
        self.listener = SyncMultiFrameListener(FrameType.Color)
        self.device.setColorFrameListener(self.listener)
        self.device.startStreams(rgb=True, depth=False)
        self._started = True

    def read_with_metadata(self, timeout_ms: int = 2000):
        """Device frame time and host receipt are different clocks; do not subtract them."""
        frames = self.listener.waitForNewFrame(milliseconds=timeout_ms)
        received = time.monotonic()
        if frames is None:
            return None
        try:
            # Color 帧是 BGRX，asarray 得到 (H, W, 4)
            frame = frames["color"]
            source_time = getattr(frame, "timestamp", None)
            metadata = {
                "receive_monotonic_s": received,
                # libfreenect2 timestamp ticks are 0.125 ms, not host time.
                "source_time_s": None if source_time is None else float(source_time) * .000125,
                "source_clock": "kinect_device_125us" if source_time is not None else "unavailable",
                "frame_id": int(getattr(frame, "sequence", -1)),
            }
            buf = frame.asarray(np.uint8)
            img = np.ascontiguousarray(buf[..., 2::-1])   # BGRX -> RGB

            if self.rotate_cw:
                img = np.ascontiguousarray(np.rot90(img, k=-1))   # 顺时针 90

            if self.crop_square:
                hh, ww = img.shape[:2]
                side = min(hh, ww)
                y0 = (hh - side) // 2
                x0 = (ww - side) // 2
                img = img[y0:y0 + side, x0:x0 + side]

            return img, metadata
        finally:
            self.listener.release(frames)

    def read(self, timeout_ms: int = 2000) -> np.ndarray | None:
        sample = self.read_with_metadata(timeout_ms)
        return None if sample is None else sample[0]

    def close(self):
        if getattr(self, "_started", False):
            try:
                self.device.stop()
                self.device.close()
            except Exception:
                pass
            self._started = False


# ---------------- RealSense D435i ----------------
class RealSenseRGB:
    """RealSense 彩色流读取器。输出 HWC uint8 RGB。"""

    def __init__(self, serial: str | None = None, width=640, height=480, fps=30):
        import pyrealsense2 as rs
        self._rs = rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipe.start(cfg)
        self.align = rs.align(rs.stream.color)
        self._started = True

    def read_with_metadata(self, timeout_ms: int = 2000):
        try:
            frames = self.pipe.wait_for_frames(timeout_ms=timeout_ms)
            received = time.monotonic()
            color = frames.get_color_frame()
            if not color:
                return None
            bgr = np.asanyarray(color.get_data())
            metadata = {
                "receive_monotonic_s": received,
                "source_time_s": float(color.get_timestamp()) * .001,
                "source_clock": "realsense_" + str(color.get_frame_timestamp_domain()),
                "frame_id": int(color.get_frame_number()),
            }
            return np.ascontiguousarray(bgr[..., ::-1]), metadata
        except Exception:
            return None

    def read(self, timeout_ms: int = 2000) -> np.ndarray | None:
        sample = self.read_with_metadata(timeout_ms)
        return None if sample is None else sample[0]

    def close(self):
        if getattr(self, "_started", False):
            try:
                self.pipe.stop()
            except Exception:
                pass
            self._started = False


# ---------------- 工具 ----------------
def resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    from PIL import Image
    cur_h, cur_w = image.shape[:2]
    ratio = max(cur_w / width, cur_h / height)
    new_w, new_h = int(round(cur_w / ratio)), int(round(cur_h / ratio))
    resized = np.asarray(Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR))
    pad_h, pad_w = height - new_h, width - new_w
    if pad_h > 0 or pad_w > 0:
        resized = np.pad(
            resized,
            ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)),
            mode="constant",
        )
    return resized


def list_devices() -> dict:
    """探测可用的相机。"""
    out = {"kinect": [], "realsense": []}
    try:
        _preload_freenect2()
        from pylibfreenect2 import Freenect2
        fn = Freenect2()
        n = fn.enumerateDevices()
        if n:
            out["kinect"].append(fn.getDefaultDeviceSerialNumber())
    except Exception as e:
        out["kinect_error"] = str(e)
    try:
        import pyrealsense2 as rs
        for d in rs.context().query_devices():
            out["realsense"].append({
                "name": d.get_info(rs.camera_info.name),
                "serial": d.get_info(rs.camera_info.serial_number),
            })
    except Exception as e:
        out["realsense_error"] = str(e)
    return out


if __name__ == "__main__":
    import json
    print("=== 相机探测 ===")
    print(json.dumps(list_devices(), indent=2, ensure_ascii=False))

    print("\n=== Kinect 试读 ===")
    try:
        k = KinectV2()
        img = k.read()
        if img is None:
            print("读取失败")
        else:
            print(f"形状={img.shape} dtype={img.dtype} 均值={img.mean():.1f} "
                  f"非零占比={100.0 * np.count_nonzero(img) / img.size:.1f}%")
            from PIL import Image
            Image.fromarray(img).resize((360, 360)).save("/tmp/kinect_wrapped.png")
            print("已保存 /tmp/kinect_wrapped.png")
        k.close()
    except Exception as e:
        print("Kinect 失败:", e)

    print("\n=== RealSense 试读 ===")
    try:
        r = RealSenseRGB()
        img = r.read()
        if img is None:
            print("读取失败")
        else:
            print(f"形状={img.shape} dtype={img.dtype} 均值={img.mean():.1f} "
                  f"非零占比={100.0 * np.count_nonzero(img) / img.size:.1f}%")
            from PIL import Image
            Image.fromarray(img).resize((360, 270)).save("/tmp/realsense_wrapped.png")
            print("已保存 /tmp/realsense_wrapped.png")
        r.close()
    except Exception as e:
        print("RealSense 失败:", e)
