import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor"))
from legacy_cameras import KinectV2, RealSenseRGB


class CameraTimestampTests(unittest.TestCase):
    def test_realsense_keeps_device_domain_and_host_receipt(self):
        camera = RealSenseRGB.__new__(RealSenseRGB)
        frame = Mock()
        frame.get_data.return_value = np.array([[[3, 2, 1]]], np.uint8)
        frame.get_timestamp.return_value = 1234500.
        frame.get_frame_number.return_value = 42
        frame.get_frame_timestamp_domain.return_value = "hardware_clock"
        camera.pipe = Mock()
        camera.pipe.wait_for_frames.return_value.get_color_frame.return_value = frame
        with patch("legacy_cameras.time.monotonic", return_value=10.):
            image, meta = camera.read_with_metadata()
        np.testing.assert_array_equal(image, [[[1, 2, 3]]])
        self.assertEqual(meta["source_time_s"], 1234.5)
        self.assertEqual(meta["receive_monotonic_s"], 10.)
        self.assertEqual(meta["source_clock"], "realsense_hardware_clock")
        self.assertEqual(meta["frame_id"], 42)

    def test_kinect_125us_units_and_frame_release(self):
        camera = KinectV2.__new__(KinectV2)
        camera.rotate_cw = camera.crop_square = False
        frame = Mock(timestamp=8000, sequence=42)
        frame.asarray.return_value = np.array([[[3, 2, 1, 0]]], np.uint8)
        camera.listener = Mock()
        frames = {"color": frame}
        camera.listener.waitForNewFrame.return_value = frames
        with patch("legacy_cameras.time.monotonic", return_value=10.):
            image, meta = camera.read_with_metadata()
        np.testing.assert_array_equal(image, [[[1, 2, 3]]])
        self.assertEqual(meta["source_time_s"], 1.)
        self.assertEqual(meta["receive_monotonic_s"], 10.)
        camera.listener.release.assert_called_once_with(frames)
