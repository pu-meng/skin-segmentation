import json, os, subprocess, sys, time
from pathlib import Path
OPS = Path('/home/PuMengYu/experiments/skin_segmentation/operations')
PYTHON = '/home/PuMengYu/anaconda3/envs/medseg/bin/python'
EXPCTL = OPS / 'expctl.py'

def update(exp_id, **fields):
    cmd = [PYTHON, str(EXPCTL), 'update', '--id', exp_id]
    for key, value in fields.items():
        if value is None: continue
        cmd += ['--' + key.replace('_', '-'), str(value)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

for seed in (43, 40):
    exp_id = f'A1_SWIFTFORMER_ES_ISIC_SEED{seed}_20261114'
    run = Path(f'/media/8T/PuMengYu/experiments/skin_segmentation/A1_SWIFTFORMER_ES_ISIC_SEED{seed}_FRESH300_20261114')
    entry = OPS / f'a1_swift_es_isic_seed{seed}_20261114' / 'entry.py'
    update(exp_id, status='waiting_gpu', wait_reason='用户指定随机种子实验空闲显存门槛4 GiB；等待GPU0满足准入后执行ISIC batch8 fresh300', pid=os.getpid())
    (run / 'ISIC2018' / 'queue_status.json').write_text(json.dumps({'status':'queued_for_gpu0','seed':seed,'updated_unix':time.time()},indent=2)+'\n')
    proc = subprocess.run([PYTHON, '-B', str(entry), 'queue', 'ISIC2018', '0'])
    if proc.returncode:
        update(exp_id, status='failed', wait_reason='训练/测试队列失败；检查运行目录failure.json')
        sys.exit(proc.returncode)
    update(exp_id, status='complete', epochs_done=300, epochs=300, gpu=0, wait_reason='', note=f'ISIC2018 seed{seed} fresh300；无TTA/TTA4测试和产物审计完成。')
