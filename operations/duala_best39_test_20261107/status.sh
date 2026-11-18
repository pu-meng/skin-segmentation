#!/bin/bash
set -eu
date -Is
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
nvidia-smi -i GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
stat '/media/8T/PuMengYu/experiments/skin_segmentation/DecoderDualA_SCREEN100of300_20261106/screen100_best39_test_20261107/stdout.log'
tail -8 '/media/8T/PuMengYu/experiments/skin_segmentation/DecoderDualA_SCREEN100of300_20261106/screen100_best39_test_20261107/stdout.log'
cat '/media/8T/PuMengYu/experiments/skin_segmentation/DecoderDualA_SCREEN100of300_20261106/screen100_best39_test_20261107/DualA/status.json'
cat '/home/PuMengYu/experiments/skin_segmentation/operations/duala_best39_test_20261107/exitcode'
stat '/media/8T/PuMengYu/experiments/skin_segmentation/DecoderDualA_SCREEN100of300_20261106/screen100_best39_test_20261107/DualA/test/summary.json' '/media/8T/PuMengYu/experiments/skin_segmentation/DecoderDualA_SCREEN100of300_20261106/screen100_best39_test_20261107/independent_artifact_audit.json'
