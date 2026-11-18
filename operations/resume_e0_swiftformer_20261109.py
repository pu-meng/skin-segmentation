"""Resume the interrupted E0 SwiftFormer-S run from its verified epoch-boundary checkpoint."""
import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('/home/PuMengYu/experiments/skin_segmentation')
RUN_SCRIPT = ROOT / 'operations/isic_e0_encoder_swap_20261108/run.py'
RUN_ROOT = Path('/media/8T/PuMengYu/experiments/skin_segmentation/ISIC2018_E0_EncoderSwap_FRESH300_20261108')
ARM = 'swiftformer_s'
GPU = 0
sys.path[:0] = [str(ROOT), str(ROOT / 'source/comparison'), str(ROOT / 'source/bottleneck_mamba')]


def main():
    spec = importlib.util.spec_from_file_location('e0_encoder_swap_run', RUN_SCRIPT)
    e0 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e0)
    e0.RUN = RUN_ROOT
    base = e0.setup_base()
    protocol = e0.verify()
    work = RUN_ROOT / ARM
    stage = work / 'run'

    import torch
    import torch.optim
    from pretrained_unet2d import prepare_imports
    prepare_imports('tinyvim_s')
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch, runner, data, build = e0.imports(ARM)
    api = runner.baseline_api()
    api.seed_all(42)
    model = build(True)
    assert e0.digest(model.state_dict()) == protocol['arms'][ARM]['initial_state_sha256']
    model.cuda()
    loaders, generator = data.loaders(protocol['config'], api)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01,
                                  betas=(.9, .999), eps=1e-8)
    criterion = api.BceDiceLoss().cuda()

    checkpoint_path = stage / 'checkpoints/latest.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    with (stage / 'history.csv').open() as stream:
        history = list(csv.DictReader(stream))
    assert checkpoint['epoch'] == len(history) == int(history[-1]['epoch']) == 264
    assert checkpoint['protocol_sha256'] == e0.sha(RUN_ROOT / 'protocol.json')
    assert checkpoint['split_sha256'] == protocol['config']['data']['split_sha256']
    assert checkpoint['best_epoch'] == 75

    # The generic recovery inspector targets an unrelated registry; validate this E0 run here.
    import recover_ham_dedup
    recover_ham_dedup.inspect_run = lambda name, config: {
        'action': 'resume', 'completed_epoch': checkpoint['epoch'],
        'next_epoch': checkpoint['epoch'] + 1,
        'protocol_sha256': checkpoint['protocol_sha256'],
        'split_sha256': checkpoint['split_sha256'],
        'checkpoint': str(checkpoint_path), 'best_epoch': checkpoint['best_epoch']}

    info = dict(model=ARM + ' ISIC2018', dataset='ISIC2018', pid=os.getpid(), seed=42,
                gpu_uuid=e0.UUIDS[GPU], physical_gpu=GPU,
                protocol_sha256=e0.sha(RUN_ROOT / 'protocol.json'),
                split_sha256=protocol['config']['data']['split_sha256'],
                source_sha256=protocol['protected_sha256'], fresh_start=True,
                training=protocol['config']['training'], test_data_accessed=False)
    (base.RUN / ARM / 'controller_status.json').write_text(json.dumps({
        'status': 'train_resume', 'controller_pid': os.getpid(), 'gpu': GPU,
        'resume_from_epoch': checkpoint['epoch'], 'updated_unix': time.time()
    }, indent=2) + '\n')

    try:
        runner.train(model, loaders, criterion, optimizer, torch.device('cuda:0'),
                     stage, protocol['config'], info, generator, resume=True)
        audit = json.loads((stage / 'artifact_audit.json').read_text())
        assert audit.get('training_complete') and audit.get('epochs') == 300
        assert audit.get('optimizer_updates') == 66900
        print('RESUME_TRAINING_COMPLETE', json.dumps(audit), flush=True)
    finally:
        for loader in loaders.values():
            if hasattr(loader, 'close'):
                loader.close()
            elif getattr(loader, '_iterator', None) is not None:
                loader._iterator._shutdown_workers()


if __name__ == '__main__':
    main()
