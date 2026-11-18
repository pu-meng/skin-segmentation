import copy,unittest
from geometry import compare_geometry
class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.a=dict(size=[512,512,600],spacing=[.64453125,.64453125,.7000000476837158],origin=[163.35546875,295.35546875,54.39996337890625],direction=[1.,0.,0.,0.,1.,0.,0.,0.,1.])
    def reject(self,b):
        with self.assertRaises(ValueError): compare_geometry(self.a,b)
    def test_real_rounding(self):
        b=copy.deepcopy(self.a);b['spacing']=[.6445313096046448,.6445313096046448,.699999988079071];b['origin'][:2]=[163.35549926757812,295.3554992675781]
        self.assertLess(compare_geometry(self.a,b)['max_corner_displacement_mm'],.001)
    def test_size(self):
        b=copy.deepcopy(self.a);b['size'][0]+=1;self.reject(b)
    def test_voxel_translation(self):
        b=copy.deepcopy(self.a);b['origin'][0]+=.64453125;self.reject(b)
    def test_cumulative_spacing_drift(self):
        b=copy.deepcopy(self.a);b['spacing'][2]+=.00001;self.reject(b)
    def test_direction_flip(self):
        b=copy.deepcopy(self.a);b['direction'][0]=-1.;self.reject(b)
    def test_nan(self):
        b=copy.deepcopy(self.a);b['origin'][0]=float('nan');self.reject(b)
if __name__=='__main__': unittest.main()
