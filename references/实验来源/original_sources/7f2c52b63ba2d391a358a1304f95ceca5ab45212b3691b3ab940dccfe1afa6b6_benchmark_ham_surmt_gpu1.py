"""The unchanged V5 efficiency workload, with full SU-RMT on physical GPU1."""
import benchmark_ham_gpu1 as device
import benchmark_ham_efficiency as benchmark
import ham_dedup_runner
from ham_surmt_runner import PROTOCOL, ROOT, MODEL, make_model, validate


def main():
    validate()
    ham_dedup_runner.PROTOCOL = PROTOCOL
    ham_dedup_runner.make_model = make_model
    benchmark.OUT = ROOT / 'runs/protocol_v5_surmt_extension_seed42/efficiency'
    benchmark.gpu0_has_foreign_work = device.gpu1_has_foreign_work
    benchmark.write_json = device.write_json
    try:
        benchmark.worker(MODEL)
    except benchmark.BenchmarkBusy as error:
        print(str(error), flush=True)
        raise SystemExit(75)


if __name__ == '__main__':
    main()
