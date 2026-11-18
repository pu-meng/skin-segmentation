"""Explicit project paths and immutable evidence for the relocated experiments."""
import copy
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
CODE = PROJECT / 'source/comparison'
CONFIGS = PROJECT / 'source/config'
RUNS = PROJECT / 'log/八模型对比/runs'
PATHS = json.loads((CONFIGS / 'paths.json').read_text())
CACHE = Path(PATHS['cache_root'])
SNAPSHOTS = json.loads((PROJECT / 'references/实验来源/source_snapshots.json').read_text())


def original_source(path):
    key = str(path)
    if key not in SNAPSHOTS:
        raise FileNotFoundError('No recorded original source: ' + key)
    return PROJECT / SNAPSHOTS[key]['file']


def original_repository_source(relative):
    suffix = '/MambaLiteUNet/' + str(relative)
    matches = [key for key in SNAPSHOTS if key.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError('Ambiguous original source: ' + str(relative))
    return original_source(matches[0])


def relocate(value):
    """Map historical provenance strings into the current layout, without links."""
    if not isinstance(value, str) or not value.startswith('/home/PuMengYu/'):
        return value
    old = '/home/PuMengYu/experiments/MambaLiteUNet/reproduction'
    materials = PATHS['materials_root']
    rules = [
        (old + '/pure_mixer_comparison/training_logs', PROJECT / 'log/Conv_Attention_Mamba_整网对比/training_logs'),
        (old + '/pure_mixer_comparison/runs', PROJECT / 'log/Conv_Attention_Mamba_整网对比/runs'),
        (old + '/pure_mixer_comparison', PROJECT / 'source/comparison_mixers'),
        (old + '/tinyvim_frequency_analysis', PROJECT / 'source/analysis/tinyvim_frequency'),
        (old + '/频谱对照试验', PROJECT / 'source/analysis/unet_frozen_frequency'),
        (old + '/data', PROJECT / 'data/processed'),
        (old + '/runs', RUNS),
    ]
    if value in PATHS['pretrained_weights'].values():
        return value
    for kind, dest in PATHS['pretrained_weights'].items():
        if Path(value).name == Path(dest).name and value.endswith(('.pth', '.pth.tar')):
            return dest
    if value.startswith(old + '/protocol') and value.endswith(('.json','.sha256')):
        return str(CONFIGS / Path(value).name)
    for name, dest in PATHS['model_sources'].items():
        if name in ('FasterViT','MambaVision','TinyViM'):
            rules.append((materials + '/' + name + '/code', Path(dest)))
    rules.extend([
        ('/home/PuMengYu/experiments/RMT/official_code', PROJECT/'references/code/RMT'),
        ('/home/PuMengYu/experiments/SU-RMT/official_code', PROJECT/'references/code/SU-RMT'),
        (old, CODE),
        ('/home/PuMengYu/experiments/MambaLiteUNet', PROJECT/'references/code/MambaLiteUNet'),
    ])
    for before, after in sorted(rules,key=lambda x:len(x[0]),reverse=True):
        if value == before or value.startswith(before + '/'):
            return str(after) + value[len(before):]
    return value


def current_protocol(path):
    def mapped(x):
        if isinstance(x,dict):return {k:mapped(v) for k,v in x.items()}
        if isinstance(x,list):return [mapped(v) for v in x]
        return relocate(x)
    return mapped(json.loads(Path(path).read_text()))


def verify_runtime():
    """A path migration has its own code manifest, separate from old protocols."""
    manifest = json.loads((PROJECT/'source/config/runtime_sources.json').read_text())
    for relative, expected in manifest.items():
        actual=hashlib.sha256((PROJECT/relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Runtime source changed; record a new revision: ' + relative)
