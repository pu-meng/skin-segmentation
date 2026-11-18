from pathlib import PurePosixPath
ROOT=PurePosixPath('/home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab')
def valid_budget(samples,batch,steps,val):
 return (samples,batch,steps,val)==(6983,8,873,1010)
def within_root(path):
 p=PurePosixPath(path)
 return '..' not in p.parts and p.is_relative_to(ROOT)
