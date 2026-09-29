# ACRS ablation recipes

Every recipe in `configs/` uses the same training data, augmentation, ArcFace
head, optimizer, learning-rate schedule, number of epochs, embedding head, and
verification protocol as the full ACRS recipe. The files differ only in the
component named by the experiment.

| Recipe | Stage 3 suppression | Stage 4 suppression | Final fusion | Age supervision | Other change |
| --- | --- | --- | --- | --- | --- |
| `acrs_resnet34_vox2.yaml` | on | on | on | on | Full ACRS |
| `acrs_resnet34_vox2_s3_only.yaml` | on | off | on | on | none |
| `acrs_resnet34_vox2_s4_only.yaml` | off | on | on | on | none |
| `acrs_resnet34_vox2_fusion_only.yaml` | off | off | on | on | none |
| `acrs_resnet34_vox2_no_age_supervision.yaml` | on | on | on | off | `lambda_age: 0` |
| `acrs_resnet34_vox2_no_age_conditioning.yaml` | off | off | off | on | conditioning hierarchy and age head retained |
| `acrs_resnet34_vox2_random_gate_init.yaml` | on | on | on | on | residual gate heads use Kaiming initialization and zero bias |
| `acrs_resnet34_vox2_no_fusion.yaml` | on | on | off | on | none |

## Stage-position ablations

`enable_acrs_s3` and `enable_acrs_s4` control only the residual subtraction at
the corresponding identity stage. A disabled stage returns the unmodified
identity tensor and a zero gate. The conditioning hierarchy, age objective,
and final condition-guided fusion remain present. The disabled module remains
in a zero-gradient graph so model parameters and checkpoint keys are identical
across the full, S3-only, S4-only, and Fusion-only systems.

## Age ablations

The no-age-supervision recipe sets only `losses.lambda_age` to zero. The
conditioning branch still affects both residual gates and final fusion, and it
can still learn through the speaker objective.

The no-age-conditioning recipe keeps the complete conditioning hierarchy and
its explicit age objective. It makes both residual suppression blocks exact
identity operations and makes the final fusion block return its identity input.
The age representation therefore remains supervised but is not routed into the
speaker embedding path.

## Gate and fusion controls

The random-gate-initialization recipe changes only the two residual gate-head
initializations. The no-fusion recipe bypasses only the final Stage-4 fusion;
both residual suppression stages remain active.

All recipe paths are repository-relative. GPU lists and per-device batch sizes
may be adapted to the local machine while keeping the global batch size at 256.
Verification uses feature-level CMVN and cosine scoring without embedding-level
mean subtraction.
