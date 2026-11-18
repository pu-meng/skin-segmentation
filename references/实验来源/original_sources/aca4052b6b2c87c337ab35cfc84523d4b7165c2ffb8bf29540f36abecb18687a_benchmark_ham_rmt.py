"""Apply the identical isolated V5 benchmark to the two RMT models."""
import argparse

import benchmark_ham_efficiency as benchmark
import ham_dedup_runner
from ham_rmt_runner import PROTOCOL, ROOT, make_model
from ham_fast_runner import validate_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['rmt_t', 'rmt_s'], required=True)
    args = parser.parse_args()
    validate_protocol(PROTOCOL, args.model)
    # Bind only this fresh subprocess to the separate protocol/model factory.
    ham_dedup_runner.PROTOCOL = PROTOCOL
    ham_dedup_runner.make_model = make_model
    benchmark.OUT = ROOT / 'runs/protocol_v5_rmt_extension_seed42/efficiency'
    try:
        benchmark.worker(args.model)
    except benchmark.BenchmarkBusy as error:
        print(str(error), flush=True)
        raise SystemExit(75)


if __name__ == '__main__':
    main()
