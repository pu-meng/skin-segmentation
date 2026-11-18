"""Unchanged V5 efficiency workload for full SU-RMT on physical GPU0."""
import benchmark_ham_efficiency as benchmark
import ham_dedup_runner
from ham_surmt_runner import PROTOCOL, ROOT, MODEL, make_model, validate


def main():
    validate()
    ham_dedup_runner.PROTOCOL = PROTOCOL
    ham_dedup_runner.make_model = make_model
    benchmark.OUT = ROOT / 'runs/protocol_v5_surmt_extension_seed42/efficiency'
    original_write = benchmark.write_json
    def write(path, value):
        original_write(path, dict(value, physical_gpu=0, logical_cuda_device=0,
            device_override='Idle GPU0 runtime scheduling override; frozen workload unchanged'))
    benchmark.write_json = write
    try:
        benchmark.worker(MODEL)
    except benchmark.BenchmarkBusy as error:
        print(str(error), flush=True)
        raise SystemExit(75)


if __name__ == '__main__':
    main()
