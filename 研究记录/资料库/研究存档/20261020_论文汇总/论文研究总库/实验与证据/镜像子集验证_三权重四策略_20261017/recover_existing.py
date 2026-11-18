"""Explicit CPU-only salvage of the failed first item; never reruns inference."""
from common import *
import fcntl

def main():
    environment('')
    with (ROOT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=read(ROOT/'queue_state.json')
        if state['status']!='blocked_item_failed' or state['current']!='BASELINE_v8':
            raise RuntimeError('Unexpected failed queue identity')
        out=ROOT/'BASELINE_v8'
        before={p.name:sha(p) for p in (out/'predictions').glob('*.nii.gz')}
        if len(before)!=13: raise RuntimeError('Recovery needs all existing predictions')
        # Original process exited before serializing timings; do not invent speed data.
        runtime=dict(prediction_seconds=None,full_item_seconds=None,sliding_window_seconds=None,
                     case_timings=[],physical_gpu=0,window_batch_size=1,full_run=True,expected=13,
                     recovered_without_inference=True,timing_available=False,
                     timing_limitation='Original prediction completed but crashed before timing serialization; speedup unavailable for BASELINE comparisons')
        from finalize import finalize_item
        finalize_item('BASELINE_v8',runtime)
        if before!={p.name:sha(p) for p in (out/'predictions').glob('*.nii.gz')}:
            raise RuntimeError('Recovery modified prediction files')
        save(out/'recovery_verification.json',dict(predictions_unchanged=True,predictions=13,cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],inference_rerun=False))
        print('RECOVERED BASELINE_v8: 13 unchanged predictions; CPU-only finalization; original timing unavailable')
if __name__=='__main__': main()
