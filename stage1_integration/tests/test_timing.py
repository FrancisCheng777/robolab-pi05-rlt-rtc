import tempfile
import unittest

import h5py
import numpy as np

from pi05_demo.episode import EpisodeWriter, inspect_episode
from pi05_demo.timing import audit_timing


class TimingTests(unittest.TestCase):
    def sample(self):
        image = np.zeros((16, 16, 3), np.uint8)
        return dict(
            q=np.zeros(7), target=np.full(7, .01), width=.04, command_width=.03,
            frames={k: (image, 9.98) for k in ("main", "wrist")},
            timestamp=10., state_sequence=1, command_sequence=1, ack=True,
            state_age=.02, gripper_age=.03,
            timing={"state_bridge_receive_s": 9.98, "state_client_receive_s": 9.99,
                    "gripper_bridge_read_s": 9.97, "gello_read_s": 10.,
                    "command_send_s": 10.001, "command_enqueue_s": 10.002,
                    "command_ack_s": 10.003},
            camera_metadata={k: {"receive_monotonic_s": 9.98,
                                 "ready_monotonic_s": 9.985,
                                 "source_time_s": 1234., "source_clock": k + "_device",
                                 "frame_id": 42} for k in ("main", "wrist")},
        )

    def test_source_clock_is_preserved_separately_from_host_and_ack(self):
        with tempfile.TemporaryDirectory() as root:
            writer = EpisodeWriter(root, mock=True, task="insert", session="test", config={})
            writer.append(**self.sample())
            path = writer.finish("success")
            inspect_episode(path, allow_mock=True)
            with h5py.File(path) as f:
                self.assertEqual(f["camera_time/main"][0], 9.98)
                self.assertEqual(f["camera_source_time/main"][0], 1234.)
                self.assertEqual(f["camera_frame_id/main"][0], 42)
                self.assertEqual(f.attrs["camera_source_clock/main"], "main_device")
                self.assertEqual(f["timing/command_ack_s"][0], 10.003)
                self.assertFalse(f.attrs["exposure_alignment_verified"])

    def test_impossible_command_timeline_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            writer = EpisodeWriter(root, mock=True, task="insert", session="test", config={})
            sample = self.sample()
            sample["timing"]["command_ack_s"] = 9.9
            try:
                with self.assertRaisesRegex(ValueError, "command.*time"):
                    writer.append(**sample)
                self.assertEqual(writer.count, 0)
            finally:
                writer.close_incomplete()

    def test_nonuniform_samples_are_not_relabelled_as_fixed_fps(self):
        with tempfile.TemporaryDirectory() as root:
            writer = EpisodeWriter(root, mock=True, task="insert", session="test", config={})
            for index, stamp in enumerate([10., 10.12, 10. + 2/15]):
                sample = self.sample()
                sample.pop("timing"); sample.pop("camera_metadata")
                sample["timestamp"] = stamp
                sample["frames"] = {k: (v[0], stamp) for k, v in sample["frames"].items()}
                sample["command_sequence"] = index + 1
                writer.append(**sample)
            path = writer.finish("success")
            with h5py.File(path) as f:
                with self.assertRaisesRegex(ValueError, "irregular control timing"):
                    audit_timing(f, 15)

    def test_legacy_source_times_are_explicitly_unavailable(self):
        with tempfile.TemporaryDirectory() as root:
            writer = EpisodeWriter(root, mock=True, task="insert", session="test", config={})
            sample = self.sample()
            sample.pop("timing"); sample.pop("camera_metadata")
            writer.append(**sample)
            path = writer.finish("success")
            with h5py.File(path) as f:
                report = audit_timing(f, 15)
                self.assertEqual(report["timing_version"], 1)
                self.assertEqual(report["cameras"]["main"]["source_clock"], "unavailable")
                self.assertFalse(f["camera_source_time_valid/main"][0])


if __name__ == "__main__":
    unittest.main()
