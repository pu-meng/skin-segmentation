"""Fetch official candidate weights with verified, resumable byte ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import requests

PROJECT = Path(__file__).resolve().parents[2]
WEIGHTS = PROJECT/'Models/pretrained/2026_candidates'
RECORDS = PROJECT/'references/核对记录/2026候选_20261104'
SOURCES = {
    'convneur_m2': ('https://github.com/ZhenyuYang01/ConvNeur/releases/download/ckpts/convneur_m2_in1k_ema.pth',
                   24546231, '7368e5ecd0f7aedb2e5134d8623806eaf75b5705e4ea1735e78512aa8a2a8eb6'),
    'microvitv2_s1': ('https://huggingface.co/novendrastywn/MicroViT/resolve/main/MicroViTv2-S1.pth', None, None),
    'locatvit_tiny': ('https://huggingface.co/sinahmr/locatvit_tiny/resolve/main/model.safetensors', None, None),
}


def download(name):
    url, size, expected = SOURCES[name]
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    RECORDS.mkdir(parents=True, exist_ok=True)
    path = WEIGHTS/(name+'.pth')
    if not size or not expected:
        r = requests.head(url, allow_redirects=False, timeout=30)
        r.raise_for_status()
        size = int(r.headers.get('X-Linked-Size') or r.headers.get('Content-Length') or 0)
        expected = r.headers.get('X-Linked-Etag', '').strip('"')
        if not size or len(expected) != 64:
            raise RuntimeError(f'Official LFS size/SHA256 unavailable: {name}: {dict(r.headers)}')
    if not path.exists():
        chunk = 2*1024*1024
        ranges = [(i, start, min(start+chunk,size)-1) for i,start in enumerate(range(0,size,chunk))]
        def part(job):
            index,start,end = job
            dest=path.with_suffix(f'.part{index}')
            if dest.exists() and dest.stat().st_size == end-start+1:
                return dest
            for attempt in range(3):
                try:
                    response=requests.get(url,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(30,90))
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                        raise RuntimeError('Server did not honor exact byte range')
                    with dest.open('wb') as stream:
                        for data in response.iter_content(256*1024): stream.write(data)
                    if dest.stat().st_size != end-start+1: raise RuntimeError('Truncated download')
                    print(name,'part',index,'complete',flush=True)
                    return dest
                except Exception:
                    if attempt == 2: raise
                    time.sleep(1)
        with ThreadPoolExecutor(max_workers=6) as pool:
            parts=list(pool.map(part,ranges))
        temporary=path.with_suffix('.assembling')
        with temporary.open('wb') as output:
            for part_path in parts:
                with part_path.open('rb') as stream:
                    while data:=stream.read(1048576): output.write(data)
        actual=hashlib.sha256(temporary.read_bytes()).hexdigest()
        if actual != expected: raise RuntimeError('Downloaded SHA256 differs from official source')
        temporary.replace(path)
        for part_path in parts: part_path.unlink()
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if path.stat().st_size != size or actual != expected: raise RuntimeError('Existing weight integrity failure')
    record={'name':name,'url':url,'path':str(path),'bytes':size,'sha256':actual,
            'official_sha256':expected,'verified':True}
    (RECORDS/(name+'_material.json')).write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('name',choices=SOURCES)
    args=parser.parse_args()
    download(args.name)
