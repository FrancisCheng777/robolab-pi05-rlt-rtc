import unittest
import numpy as np
from pi05_demo.contract import encode, decode, MAX_WIDTH

class ContractTests(unittest.TestCase):
    def test_physical_target_round_trip(self):
        q = np.array([0.,-.4,0.,-2.,0.,1.5,0.])
        target=q+np.array([.01,-.02,0.,.005,0.,0.,0.])
        a=encode(q,target,.04)
        recovered,width=decode(q,a)
        np.testing.assert_allclose(recovered,target,atol=1e-8)
        self.assertAlmostEqual(width,.04,places=7)

    def test_open_and_closed_have_explicit_sign(self):
        q=np.zeros(7)
        self.assertEqual(encode(q,q,0.)[7],-1)
        self.assertEqual(encode(q,q,MAX_WIDTH)[7],1)

    def test_unrepresentable_command_is_rejected_not_clipped(self):
        with self.assertRaises(ValueError):
            encode(np.zeros(7),np.full(7,.031),.04)

    def test_nonfinite_command_is_rejected(self):
        with self.assertRaises(ValueError):
            encode(np.zeros(7),np.full(7,float("nan")),.04)
