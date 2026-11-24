# SwiftFuse

SwiftFuse is a skin lesion segmentation network with a SwiftFormer-S encoder, an SSB bottleneck, DWB and DualStage decoder blocks, and a DBCS skip connection. The original experiment code uses the name A1-SwiftFormer.

- [Reproduction instructions](reproducibility/a1_swiftformer/README.md)
- [Recorded environment](reproducibility/a1_swiftformer/requirements-original.txt)
- [Fixed split and checkpoint checksums](reproducibility/a1_swiftformer/evaluation.json)
- [AI reproduction handoff](AI_REPRODUCE.md)

The portable workflow supports HAM10000 preprocessing and training, and TTA4 evaluation on HAM10000, ISIC2018, PH², and ISIC2017 with verified prepared inputs and a validation-selected checkpoint. The reproduction instructions distinguish the checks performed from those pending on a clean CUDA system. Dataset images, masks, and model weights are obtained separately.
