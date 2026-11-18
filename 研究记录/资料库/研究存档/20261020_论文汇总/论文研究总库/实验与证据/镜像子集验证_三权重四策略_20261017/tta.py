import torch
from pumengyu.moe_sparse.window_batch import WindowBatchPredictor
VIEWS={1:((),),2:((),(2,)),4:((),(0,),(1,),(2,)),8:((),(0,),(1,),(2,),(0,1),(0,2),(1,2),(0,1,2))}
class SubsetPredictor(WindowBatchPredictor):
    def __init__(self,*args,views=8,**kwargs):
        super().__init__(*args,window_batch_size=1,**kwargs)
        if views not in VIEWS: raise ValueError(views)
        self.views=views
    @torch.inference_mode()
    def _internal_maybe_mirror_and_predict(self,x):
        assert tuple(self.allowed_mirroring_axes)==(0,1,2)
        prediction=self.network(x)
        for axes in VIEWS[self.views][1:]:
            dims=tuple(a+2 for a in axes)
            prediction+=torch.flip(self.network(torch.flip(x,dims)),dims)
        prediction/=self.views
        return prediction

    @torch.inference_mode()
    def predict_sliding_window_return_logits(self,input_image):
        # Same padding, slicers, autocast and accumulation as the frozen path,
        # but propagate failure instead of retrying with CPU accumulation.
        from time import perf_counter
        from contextlib import nullcontext
        from acvl_utils.cropping_and_padding.padding import pad_nd_image
        from nnunetv2.utilities.helpers import empty_cache
        self.network=self.network.to(self.device); self.network.eval()
        empty_cache(self.device)
        if self.device.type=='cuda':
            torch.cuda.synchronize(self.device); torch.cuda.reset_peak_memory_stats(self.device)
        started=perf_counter(); self.tile_attempts=[]
        with torch.autocast(self.device.type) if self.device.type=='cuda' else nullcontext():
            assert input_image.ndim==4
            data,revert=pad_nd_image(input_image,self.configuration_manager.patch_size,'constant',{'value':0},True,None)
            slicers=self._internal_get_sliding_window_slicers(data.shape[1:])
            result=self._internal_predict_sliding_window_return_logits(data,slicers,self.perform_everything_on_device)
            result=result[(slice(None),*revert[1:])]
        if self.device.type=='cuda': torch.cuda.synchronize(self.device)
        self.case_timings.append(dict(case_index=len(self.case_timings),shape=list(input_image.shape),tile_attempts=self.tile_attempts,sliding_window_seconds=perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device) if self.device.type=='cuda' else None,peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device) if self.device.type=='cuda' else None))
        empty_cache(self.device)
        return result
