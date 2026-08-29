# ACRS

Official PyTorch implementation of **Age-Conditioned Residual Suppression
(ACRS)** for cross-age speaker verification.

ACRS separates the upper stages of a ResNet34 speaker encoder into identity
and age streams. Age-conditioned residual blocks suppress age-related
components in the identity stream, and a bounded fusion gate combines the two
streams before attentive spatiotemporal statistics pooling. Age labels are
used only as auxiliary supervision during training; inference requires only
acoustic features.

## Repository scope

This repository contains the standalone ACRS method, its public configuration,
and tests. It intentionally excludes unrelated architectures, private data,
experiment outputs, checkpoints, and internal research tooling. The ResNet
input convention follows [WeSpeaker](https://github.com/wenet-e2e/wespeaker):
`[batch, time, feature]`.

## Installation

```bash
git clone https://github.com/pzj08/ACRS.git
cd ACRS
python -m pip install -e '.[test]'
```

## Minimal example

```python
import torch
from acrs import ACRS_ResNet34

model = ACRS_ResNet34(
    feat_dim=80,
    embed_dim=256,
    acrs_args={
        "age_bins": [18, 25, 35, 45, 55, 65],
        "losses": {"lambda_age": 0.05},
    },
)

features = torch.randn(4, 200, 80)
output = model(features)
embeddings = output["embedding"]  # [4, 256], L2-normalized
```

During training, discretize each utterance's age with the same `age_bins` and
pass the resulting group index to `compute_acrs_losses`:

```python
age_groups = torch.tensor([1, 3, 5, 6])
output = model(features, age_groups=age_groups)
aux = model.compute_acrs_losses(output, age_groups, epoch=1)
loss = speaker_classification_loss + aux["loss_acrs_total"]
```

Unknown ages should use `ignore_age_index` (default: `-1`). See
[`configs/acrs_resnet34.yaml`](configs/acrs_resnet34.yaml) for the complete
configuration.

## Ablations

Set `acrs_args.ablation.mode` to one of:

- `no_age_conditioning`: disable residual suppression and the fusion gate.
- `no_residual_suppression`: disable the two residual-suppression gates.
- `no_fusion_gate`: disable only the cross-stream fusion gate.
- `random_gate_init`: replace the retention-preserving gate initialization.

The default is `none`.

## Testing

```bash
pytest -q
```

The tests cover inference without age labels, auxiliary-loss backpropagation,
counterfactual training, and every public ablation.

## WeSpeaker integration

Copy `acrs/` into the Python environment used by WeSpeaker, register
`ACRS_ResNet34` in `wespeaker.models.speaker_model`, and select
`output["embedding"]` wherever the training or extraction pipeline consumes a
speaker embedding. Add `loss_acrs_total` to the primary ArcFace speaker loss.
The factory accepts WeSpeaker's `pooling_func` and `two_emb_layer` arguments for
configuration compatibility.

## Citation

Citation information will be added with the paper release.

## Acknowledgements

The ResNet acoustic-encoder convention is based on WeSpeaker. This repository
is released under the Apache License 2.0; see [LICENSE](LICENSE).
