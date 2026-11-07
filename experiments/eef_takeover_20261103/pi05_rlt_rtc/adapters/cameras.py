"""Camera interface with nonblocking latest-frame reads on the controller thread."""
from __future__ import annotations

import dataclasses
import importlib
import threading
import time
from typing import Protocol

import numpy as np

from pi05_rlt_rtc.contracts.errors import ContractError, EnvironmentNotReadyError, MissingAssetError


@dataclasses.dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray
    captured_monotonic: float  # same host monotonic clock as the Python controller
    metadata: dict = dataclasses.field(default_factory=dict)


class CameraProvider(Protocol):
    def capture(self) -> dict[str, CameraFrame]: ...
    def close(self) -> None: ...


def validate_frames(frames, *, now, max_age, max_skew):
    times = []
    for key in ("wrist",):
        if key not in frames:
            raise ContractError(f"missing {key} camera")
        frame = frames[key]
        pixels = np.asarray(frame.rgb)
        if pixels.ndim != 3 or pixels.shape[-1] != 3 or pixels.dtype != np.uint8:
            raise ContractError(f"{key} must provide HWC uint8 RGB")
        age = now - frame.captured_monotonic
        if not np.isfinite(age) or age < 0 or age > max_age:
            raise ContractError(f"stale or invalid {key} camera timestamp: age={age}")
        times.append(frame.captured_monotonic)
    if max(times) - min(times) > max_skew:
        raise ContractError("main/wrist camera skew exceeds configured limit")


class OpenCVCameras:
    """USB/V4L2 sources. Timestamp is host receive time, not sensor exposure time.

    Supply a CameraProvider factory for RealSense/ROS hardware timestamps or
    preprocessing/cropping used by the training dataset.
    """
    def __init__(self, config):
        for key in ("main", "wrist"):
            if config.get(key, {}).get("source") is None:
                raise MissingAssetError(f"configure franka.cameras.{key}.source")
        try:
            import cv2
        except ImportError as exc:
            raise EnvironmentNotReadyError("install pi05-rlt-rtc[cameras] on the controller host") from exc
        self._cv2, self._frames, self._caps, self._threads = cv2, {}, [], []
        self._lock, self._stop = threading.Lock(), threading.Event()
        try:
            for key in ("main", "wrist"):
                c = config[key]
                cap = cv2.VideoCapture(c["source"])
                self._caps.append(cap)
                if not cap.isOpened():
                    raise EnvironmentNotReadyError(f"cannot open {key} camera {c['source']!r}")
                for prop, name in ((cv2.CAP_PROP_FRAME_WIDTH, "width"), (cv2.CAP_PROP_FRAME_HEIGHT, "height"),
                                   (cv2.CAP_PROP_FPS, "fps")):
                    if name in c:
                        cap.set(prop, c[name])
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                thread = threading.Thread(target=self._read, args=(key, cap), daemon=True)
                thread.start()
                self._threads.append(thread)
            deadline = time.monotonic() + 5.
            while time.monotonic() < deadline:
                with self._lock:
                    if len(self._frames) == 2:
                        break
                time.sleep(.01)
            else:
                raise EnvironmentNotReadyError("timed out waiting for both camera streams")
        except BaseException:
            self.close()
            raise

    def _read(self, key, cap):
        while not self._stop.is_set():
            ok, bgr = cap.read()
            if not ok:
                self._stop.wait(.01)
                continue
            frame = CameraFrame(self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB), time.monotonic())
            with self._lock:
                self._frames[key] = frame

    def capture(self):
        with self._lock:
            return {k: CameraFrame(v.rgb.copy(), v.captured_monotonic) for k, v in self._frames.items()}

    def close(self):
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=.2)
        for cap in self._caps:
            cap.release()


def create_cameras(config):
    if "factory" in config:
        module, name = config["factory"].split(":", 1)
        return getattr(importlib.import_module(module), name)(**config.get("kwargs", {}))
    return OpenCVCameras(config)
