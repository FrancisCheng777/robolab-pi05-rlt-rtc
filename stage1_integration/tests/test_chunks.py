import unittest
import numpy as np
from pi05_demo.chunks import complete_chunk_starts


class ChunkTests(unittest.TestCase):
    def test_windows_do_not_cross_boundaries_and_keep_final_real_action(self):
        episodes = [0] * 12 + [1] * 4 + [2] * 11
        starts = complete_chunk_starts(episodes, 10, timestamps=np.arange(len(episodes))/15)
        np.testing.assert_array_equal(starts, [0, 1, 2, 16, 17])
        actions = np.arange(len(episodes))
        np.testing.assert_array_equal(actions[starts[2]:starts[2] + 10], np.arange(2, 12))
        for start in starts:
            self.assertEqual(len(set(episodes[start:start + 10])), 1)

    def test_short_episodes_are_not_padded(self):
        with self.assertRaisesRegex(ValueError, "no complete action chunks"):
            complete_chunk_starts([0] * 9 + [1] * 9, 10, timestamps=np.arange(18)/15)

    def test_one_step_windows_keep_every_frame(self):
        np.testing.assert_array_equal(complete_chunk_starts([0, 0, 1], 1, timestamps=np.arange(3)/15), [0, 1, 2])
