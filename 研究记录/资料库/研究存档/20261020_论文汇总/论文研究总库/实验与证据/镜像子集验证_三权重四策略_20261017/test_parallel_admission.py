import unittest
from admission import resource_ready
class ParallelAdmission(unittest.TestCase):
    def setUp(self):
        self.sample=dict(compute_pids=[101],available_ram_gib=30,memory_free_mib=16000,memory_used_mib=8000,utilization_pct=100)
    def test_share_with_one_owned_formal(self):
        self.assertTrue(resource_ready(self.sample,[101],True))
    def test_two_formal_startups_allow_mirror(self):
        self.assertTrue(resource_ready(self.sample,[101,102],True))
        self.assertFalse(resource_ready(self.sample,[101,102,103],True))
    def test_other_user_task_is_accounted_by_free_budget(self):
        self.assertTrue(resource_ready(self.sample,[102],True))
    def test_absolute_free_memory_required(self):
        self.sample['memory_free_mib']=8000
        self.assertFalse(resource_ready(self.sample,[101],True))
    def test_available_ram_required(self):
        self.sample['available_ram_gib']=11
        self.assertFalse(resource_ready(self.sample,[101],True))
    def test_old_exclusive_rule_still_available(self):
        self.assertFalse(resource_ready(self.sample,parallel=False))
if __name__=='__main__':unittest.main()
