import unittest
import numpy as np
from action_adapter import convert,gripper_state

class ContractTests(unittest.TestCase):
    def test_droid_small_velocity_and_gripper_direction(self):
        command,audit=convert([.1]*7+[0])
        np.testing.assert_allclose(command[:7]*.05,.02)
        self.assertEqual(command[7],1)
        self.assertEqual(convert([0]*7+[1])[0][7],-1)
        self.assertEqual(gripper_state(.085)[0],0)
        self.assertEqual(gripper_state(0)[0],1)

    def test_large_actions_stay_within_existing_station_bound(self):
        command,audit=convert([1,-1,.3,-.3,0,.1,-.1,.6])
        self.assertLessEqual(np.max(np.abs(command)),1)
        self.assertTrue(audit['station_delta_limited'])
        np.testing.assert_allclose(command[:2]*.05,[.05,-.05])

    def test_invalid_action_never_becomes_command(self):
        for value in [[0]*7,[float('nan')]*8,[float('inf')]*8]:
            with self.assertRaises(ValueError):convert(value)
        with self.assertRaises(ValueError):gripper_state(-.1)

if __name__=='__main__':unittest.main()
