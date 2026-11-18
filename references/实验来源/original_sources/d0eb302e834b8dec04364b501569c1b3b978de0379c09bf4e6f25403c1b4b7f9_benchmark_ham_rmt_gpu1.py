"""Run the locked RMT efficiency measurement on physical GPU1."""
import benchmark_ham_gpu1 as device
import benchmark_ham_rmt as rmt

if __name__ == '__main__':
    device.benchmark.gpu0_has_foreign_work = device.gpu1_has_foreign_work
    device.benchmark.write_json = device.write_json
    rmt.main()
