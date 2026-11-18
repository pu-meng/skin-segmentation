"""Retry the frozen HAM test phase with the ablation decoder's actual head path."""
import fcntl
import sys
from pathlib import Path

ROOT = Path('/home/PuMengYu/experiments/skin_segmentation')
OPS = Path(__file__).resolve().parent
sys.path.insert(0, str(OPS))

import entry

import torch
from torch.nn import functional as F
from operations.nnunet_mirror_validation.mirror_core import mirror_logits
import operations.nnunet_mirror_validation.run_validation as validation


class AblationPredictor(torch.nn.Module):
    """Capture AblDecoder.base.head logits while preserving its forward contract."""

    def __init__(self, model, tta):
        super().__init__()
        self.model = model
        self.tta = tta
        self.times = []
        self.verify_remaining = 4
        self.max_error = 0.0
        self.raw = None
        decoder = model.decoder
        head_owner = getattr(decoder, 'base', decoder)
        self.handle = head_owner.head.register_forward_hook(self.capture)

    def capture(self, _module, _inputs, output):
        self.raw = output

    def logits(self, x):
        self.raw = None
        probability = self.model(x)
        assert self.raw is not None
        logits = F.interpolate(self.raw, size=(256, 256), mode='bilinear', align_corners=False)
        if self.verify_remaining:
            error = float((torch.sigmoid(logits) - probability).abs().max())
            self.max_error = max(self.max_error, error)
            torch.testing.assert_close(torch.sigmoid(logits), probability, rtol=0, atol=0)
            self.verify_remaining -= 1
        return logits

    def forward(self, x):
        import time
        torch.cuda.synchronize()
        started = time.perf_counter()
        probability = torch.sigmoid(mirror_logits(self.logits, x, self.tta))
        torch.cuda.synchronize()
        self.times.append(time.perf_counter() - started)
        if len(self.times) % 100 == 0:
            print('FORWARD', len(self.times), 'mirror4' if self.tta else 'none', flush=True)
        return probability

    def close(self):
        self.handle.remove()


def main():
    module = validation
    module.Predictor = AblationPredictor
    work = entry.RUN / 'HAM10000'
    # The first attempt stopped before inference, leaving only this empty directory.
    none_dir = work / 'final_test' / 'none'
    if none_dir.exists():
        assert none_dir.is_dir() and not any(none_dir.iterdir()), f'nonempty partial output: {none_dir}'
        none_dir.rmdir()

    with (work / 'controller.lock').open('a') as controller_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gate_path = ROOT / 'operations/encoder_swap_3slot_takeover_20261108/admission.lock'
        with gate_path.open('a') as gate:
            fcntl.flock(gate, fcntl.LOCK_EX)
            ready, pids, free_mib = entry.resource_ready(1)
            assert ready, f'GPU1 resource gate failed: pids={pids}, free_mib={free_mib}'
            print('GPU1_GATE_PASS', {'compute_pids': pids, 'free_mib': free_mib}, flush=True)
            entry.verify()
            entry.write(work / 'controller_status.json', {
                'status': 'test_retry', 'pid': __import__('os').getpid(),
                'physical_gpu': 1, 'reason': 'predictor head path adapted for AblDecoder',
            })
            entry.phase('HAM10000', 'test', 1)
            fcntl.flock(gate, fcntl.LOCK_UN)
    print('TEST_RETRY_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
