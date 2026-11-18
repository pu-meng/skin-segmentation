# Guidance for AI coding assistants

For requests to reproduce the main A1-SwiftFormer model, first read `AI_REPRODUCE.md` and `reproducibility/a1_swiftformer/README.md`. Inspect the code and actual local environment; do not assume historical server paths exist. This guidance is also intended to be pasted into assistants that do not read `AGENTS.md` automatically.

The documented frozen baseline is `swiftformer_s__A1`, seed 42 on HAM10000. Other A1-SwiftFormer variants are separate experiments; verify the requested variant before claiming an exact reproduction. Do not silently change data split, pretrained initialization, model, training protocol, evaluation or batch size. Distinguish model initialization, one-batch testing, full training and final evaluation, and report only checks actually completed. See `AI_REPRODUCE.md` for required files, commands, verification criteria and known gaps.
