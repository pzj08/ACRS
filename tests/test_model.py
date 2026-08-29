import pytest
import torch

from acrs import ACRS_ResNet34


def _model(**overrides):
    config = {
        "age_bins": [18, 25, 35, 45, 55, 65],
        "losses": {"lambda_age": 0.05},
    }
    config.update(overrides)
    return ACRS_ResNet34(feat_dim=80, embed_dim=64, acrs_args=config)


def test_forward_and_inference_without_age_labels():
    model = _model().eval()
    with torch.no_grad():
        output = model(torch.randn(2, 120, 80))
    assert output["embedding"].shape == (2, 64)
    assert output["age_prediction"].shape == (2,)
    assert output["age_logits"].shape == (2, 4)
    assert torch.allclose(output["embedding"].norm(dim=1), torch.ones(2),
                          atol=1.0e-5)


def test_auxiliary_losses_and_backward():
    model = _model(losses={
        "lambda_age": 0.05,
        "lambda_consistency": 0.02,
        "lambda_path": 0.01,
        "training_counterfactual": True,
    })
    classifier = torch.nn.Linear(64, 8)
    features = torch.randn(4, 120, 80)
    speakers = torch.tensor([0, 1, 2, 3])
    ages = torch.tensor([0, 2, 4, 6])
    output = model(features, age_groups=ages)
    losses = model.compute_acrs_losses(
        output, ages, speakers=speakers,
        speaker_classifier=classifier, epoch=2)
    primary = torch.nn.functional.cross_entropy(
        classifier(output["embedding"]), speakers)
    (primary + losses["loss_acrs_total"]).backward()
    assert losses["loss_acrs_total"].item() > 0
    assert all(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize(
    "mode,residual_disabled,fusion_disabled",
    [
        ("none", False, False),
        ("no_age_conditioning", True, True),
        ("no_residual_suppression", True, False),
        ("no_fusion_gate", False, True),
        ("random_gate_init", False, False),
    ],
)
def test_ablation_routing(mode, residual_disabled, fusion_disabled):
    model = _model(ablation={"mode": mode})
    assert model.residual3.disabled is residual_disabled
    assert model.residual4.disabled is residual_disabled
    assert model.fusion.disabled is fusion_disabled
    if mode == "random_gate_init":
        assert torch.count_nonzero(model.residual3.gate.bias) == 0
        assert torch.count_nonzero(model.residual4.gate.bias) == 0


def test_rejects_invalid_age_bins_and_ablation():
    with pytest.raises(ValueError, match="strictly increasing"):
        _model(age_bins=[25, 18])
    with pytest.raises(ValueError, match="unknown ablation"):
        _model(ablation={"mode": "unknown"})
