import unittest
from scheduling import decide
class QueueTests(unittest.TestCase):
 def test_missing_upstream_waits(self):self.assertFalse(decide(['waiting','complete']))
 def test_both_complete_release(self):self.assertTrue(decide(['complete','complete']))
 def test_failure_blocks(self):
  with self.assertRaises(RuntimeError):decide(['failed','complete'])
 def test_phase_gap_waits(self):self.assertFalse(decide(['test','complete']))
if __name__=='__main__':unittest.main()
