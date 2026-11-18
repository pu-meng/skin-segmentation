import os,sys,importlib.util,json,traceback,fcntl
from pathlib import Path
P=Path('/home/PuMengYu/experiments/skin_segmentation'); OPS=P/'operations/duala_E2_batch1_20261107'
sys.path[:0]=[str(OPS),str(P)]
spec=importlib.util.spec_from_file_location('original_duala_entry',OPS/'entry.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.RUN=Path('/media/8T/PuMengYu/experiments/skin_segmentation/DualA_E2_Cross_AFNO_GPU1_20261107');m.BATCHES=[['Cross','AFNO']]
original_write=m.write
def gpu1_write(path,value):
 if Path(path).name=='provenance.json':value=dict(value,physical_gpu=1)
 return original_write(path,value)
m.write=gpu1_write
if __name__ == '__main__':
 arm=sys.argv[1]
 with (m.RUN/arm/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:m.train(arm)
  except BaseException as e:
   m.write(m.RUN/arm/'failure.json',{'error':repr(e),'pid':os.getpid()});raise
