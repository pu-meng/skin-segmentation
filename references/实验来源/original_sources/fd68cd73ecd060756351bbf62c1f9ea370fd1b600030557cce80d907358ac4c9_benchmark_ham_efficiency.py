"""Queue isolated speed/memory benchmarks after the five formal runs finish."""
import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'runs/protocol_v5_dedup_seed42'
OUT = RUNS / 'efficiency'
MODELS = ['unet2d', 'mambalite', 'fastervit0', 'mambavision_t', 'tinyvim_s']


class BenchmarkBusy(RuntimeError):
    pass


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def gpu0_has_foreign_work():
    uuid = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=uuid',
                                    '--format=csv,noheader'], text=True).strip()
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                         '--format=csv,noheader,nounits'], text=True)
    for line in processes.splitlines():
        parts = [x.strip() for x in line.split(',')]
        if len(parts) == 2 and parts[1] == uuid and int(parts[0]) != os.getpid():
            return True
    return False


def worker(name):
    if gpu0_has_foreign_work():
        raise BenchmarkBusy('GPU0 is busy; defer isolated benchmark')
    import statistics
    import psutil
    import torch
    from ham_dedup_runner import PROTOCOL, baseline_api, make_model
    from pretrained_unet2d import sha256_file
    config = json.loads(PROTOCOL.read_text())
    api = baseline_api()
    api.seed_all(42)
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    model = make_model(name, config)
    checkpoint = Path(config['models'][name]['run_dir']) / 'checkpoints/best.pth'
    state = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(state, strict=True)
    del state
    model = model.to('cuda:0').eval()
    x = torch.rand(1, 3, 256, 256, device='cuda:0')
    with torch.no_grad():
        for _ in range(20):
            model(x)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        timings = []
        for _ in range(100):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            model(x)
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end))
    mean_ms = statistics.mean(timings)
    inference_allocated = torch.cuda.max_memory_allocated() / 2**20
    inference_reserved = torch.cuda.max_memory_reserved() / 2**20
    inference_pss = psutil.Process().memory_full_info().pss / 2**20
    del x
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = api.BceDiceLoss().to('cuda:0')
    x = torch.rand(8, 3, 256, 256, device='cuda:0')
    y = torch.zeros(8, 1, 256, 256, device='cuda:0')
    def step():
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss on disposable memory-benchmark copy')
        loss.backward()
        optimizer.step()
    for _ in range(3):
        step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    result = {'model': name, 'status': 'complete', 'gpu': torch.cuda.get_device_name(),
              'torch': torch.__version__, 'dtype': 'float32', 'amp': False, 'cpu_threads': 4,
              'checkpoint_sha256': sha256_file(checkpoint), 'checkpoint_size_mib': checkpoint.stat().st_size / 2**20,
              'input': [1, 3, 256, 256], 'synthetic_inputs_no_dataset_labels': True,
              'latency_scope': 'GPU-resident model forward only, excluding load/H2D/postprocessing',
              'warmup': 20, 'repeats': 100, 'mean_inference_ms': mean_ms,
              'median_inference_ms': statistics.median(timings),
              'p95_inference_ms': sorted(timings)[94], 'images_per_second_batch1': 1000 / mean_ms,
              'inference_peak_allocated_mib': inference_allocated,
              'inference_peak_reserved_mib': inference_reserved,
              'inference_steady_pss_mib': inference_pss,
              'training_memory_batch': 8, 'training_memory_optimizer': 'AdamW, initialized states',
              'training_peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
              'training_peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
              'training_steady_pss_mib': psutil.Process().memory_full_info().pss / 2**20,
              'process_peak_rss_mib_including_imports_initialization': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              'training_steps_on_disposable_copy_only': 6}
    if gpu0_has_foreign_work():
        raise BenchmarkBusy('Another compute process appeared during benchmark; measurement discarded')
    write_json(OUT / 'isolated' / f'{name}.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS)
    args = parser.parse_args()
    if args.model:
        try:
            worker(args.model)
        except BenchmarkBusy as error:
            print(str(error), flush=True)
            sys.exit(75)
        return
    config = json.loads((ROOT / 'protocol_v5_dedup_seed42.json').read_text())
    lock = OUT / 'BENCHMARK_QUEUE_RUNNING.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()) + '\n')
    try:
        while True:
            complete = all((Path(m['run_dir']) / 'artifact_audit.json').exists()
                           and json.loads((Path(m['run_dir']) / 'artifact_audit.json').read_text()).get('validation_artifacts_complete')
                           for m in config['models'].values())
            write_json(OUT / 'benchmark_queue_status.json', {'status': 'waiting_for_idle_gpu' if complete else 'waiting_for_training_completion',
                       'pid': os.getpid(), 'gpu': 0, 'models': MODELS, 'updated_unix': time.time()})
            if complete and not gpu0_has_foreign_work():
                break
            time.sleep(30)
        rows = []
        for name in MODELS:
            while gpu0_has_foreign_work():
                time.sleep(30)
            write_json(OUT / 'benchmark_queue_status.json', {'status': 'benchmarking', 'pid': os.getpid(), 'model': name})
            log = OUT / 'isolated' / f'{name}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            while True:
                while gpu0_has_foreign_work():
                    time.sleep(30)
                with log.open('a') as stream:
                    result = subprocess.run([sys.executable, '-u', __file__, '--model', name], cwd=ROOT,
                                            stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode == 75:
                    time.sleep(30)
                    continue
                if result.returncode:
                    raise RuntimeError(f'{name} benchmark exited {result.returncode}; see {log}')
                break
            rows.append(json.loads((OUT / 'isolated' / f'{name}.json').read_text()))
            write_json(OUT / 'isolated_benchmarks.json', {'models': rows})
        import csv
        with (OUT / 'isolated_benchmarks.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(OUT / 'benchmark_queue_status.json', {'status': 'complete', 'pid': os.getpid(), 'models': MODELS})
    except BaseException as error:
        write_json(OUT / 'benchmark_queue_status.json', {'status': 'failed', 'pid': os.getpid(), 'error': repr(error)})
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
