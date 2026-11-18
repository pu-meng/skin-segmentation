# Give this file to your AI assistant: A1-SwiftFormer reproduction

This document works with ChatGPT, DeepSeek, and coding agents. A web chatbot cannot access a local computer or GitHub unless the user gives it the relevant files or a working connection. A terminal/IDE coding agent can inspect and run commands only when its environment and permissions allow it.

## Suggested prompt

> I want to reproduce this repository's original A1-SwiftFormer experiment. First read `AI_REPRODUCE.md`, `AGENTS.md`, and `reproducibility/a1_swiftformer/README.md`. Inspect the actual code, environment, GPU, data and weights. Tell me what is already available and what is missing. Help me obtain the missing authorized materials, verify the split and hashes, and run the documented checks. Follow the recorded protocol and do not silently substitute data, weights, model variants or hyperparameters. Report actual command results and distinguish a model check, a batch test, completed training and final evaluation. If a step is missing or fails, explain the evidence and work toward fixing it; do not claim reproduction until the full procedure and results have actually been verified.

For chat-only use, upload the repository ZIP if supported; otherwise provide the files identified below and the relevant error output. For an agent with filesystem access, point it at the extracted repository root. The dataset, model checkpoint and private machine paths are not included in the ZIP.

## What is being reproduced

- Model: `swiftformer_s__A1`, SwiftFormer-S encoder with SSB, DBCS, DualStage and DWB; this is the original complete A1 rather than E+S, E+S+D, v0.4 or other experiments. If the requested result concerns a different experiment, trace that experiment separately.
- Task: HAM10000 binary segmentation, seed 42. Frozen split contains 6,983 training, 1,010 validation and 2,020 test images. Test data must not be used for model selection.
- Source of truth: `reproducibility/a1_swiftformer/run.py`, `protocol.json`, `split.csv`; original experiment builder is `operations/encoder_swap_A016_20261107/encoder_swap_candidate.py`, with model implementations under `source/e123_candidate_a_20261106/`, `source/e123_joint_20261105/`, `source/decoder_dual_a_20261106/` and `source/three_module_impl_20261104/`. The retained SwiftFormer classifier source is `references/code/SwiftFormer/models/swiftformer.py`.
- Reference initialization: 9,843,668 parameters; SHA256 over sorted names and raw tensor bytes is `110fd8517cb98b3925c4d7261c173b9fe000f603812a6220be180e733e6c5f4d`. `model-check` verifies this after strict pretrained loading. This is a model initialization check, not a training result.

## Prerequisites and checks

1. Inspect Python, PyTorch, CUDA/GPU and available space. The README lists versions observed on the original machine. These are reference versions, not a tested fresh installation recipe. Install CUDA extensions only after establishing compatibility with the user's system.
2. Obtain the official SwiftFormer-S classifier weights through the link in the README, put them at `Models/pretrained/tinyvim_peers_20261101/swiftformer_s.pth` and verify SHA256 `f8f93ffd5fd53e9086806f31b2c1577cba4b6eb2eb66ff20a7444e294ae5a4b5`. Do not upload weights or data to a third-party service without the user's instruction.
3. Download the four official HAM10000 files listed in the README, and run `python reproducibility/a1_swiftformer/prepare_ham.py --source /path/to/HAM10000 --output /path/to/new-cache`. Alternatively obtain the original caches securely. The script verifies split indices and SHA256; a full rebuild from the original archives has matched both original hashes in the original environment. `run.py train` verifies the cache again. The source images are not bundled. If hashes differ, investigate source files and Pillow version; do not call different preprocessing or a changed split an exact reproduction.
4. In the repository root run `python reproducibility/a1_swiftformer/run.py model-check --cache /path/to/cache`. This mode does not read the cache, but requires the official weights and compatible installed packages. Save actual output.
5. After verifying materials, availability and resource use with the user, run `python reproducibility/a1_swiftformer/run.py train --cache /path/to/cache --output /path/to/new-run`. The output directory must not exist. Recorded training: 300 epochs, batch 8, AdamW, BCE+Dice, cosine scheduler T_max 50, choose the validation-loss best. Do not start a long run merely because this file exists; follow the user's current instructions.
6. Verify outputs and logs: epoch count, optimizer update count, best checkpoint and validation result. Full final test evaluation (none/mirror4) is not yet exposed in this portable entry. Investigate and implement or port it faithfully before claiming final test reproduction; audit against the original evaluation implementation and record exactly which checks passed.

## Known verification boundaries

The portable entry has passed original-environment model construction, exact initial-state hash verification, a training-data batch read, and an exact-hash reconstruction of both HAM10000 caches from the original archives. A clean installation, GPU forward/backward through this entry, complete 300-epoch run, and final none/mirror4 evaluation have not been verified. Different GPU/software versions may not be bitwise deterministic; report versions and actual metrics. A dataset reconstruction, different preprocessing, different seed or non-identical weights is a separate experimental condition and must be labelled as such. Never invent performance numbers or claim unseen runs completed.
