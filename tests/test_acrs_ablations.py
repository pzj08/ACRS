from copy import deepcopy
from pathlib import Path

import torch
import yaml

from acrs.model import ACRS


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def load_config(name):
    with (CONFIG_DIR / name).open() as stream:
        return yaml.safe_load(stream)


def acrs_args(name):
    return load_config(name)["model_args"]["acrs_args"]


def test_recipe_differences_are_component_local():
    base = load_config("acrs_resnet34_vox2.yaml")
    expected = {
        "acrs_resnet34_vox2_s3_only.yaml": {
            "enable_acrs_s4": False,
        },
        "acrs_resnet34_vox2_s4_only.yaml": {
            "enable_acrs_s3": False,
        },
        "acrs_resnet34_vox2_fusion_only.yaml": {
            "enable_acrs_s3": False,
            "enable_acrs_s4": False,
        },
        "acrs_resnet34_vox2_no_age_supervision.yaml": {
            "losses.lambda_age": 0.0,
        },
        "acrs_resnet34_vox2_no_age_conditioning.yaml": {
            "ablation.mode": "no_age_conditioning",
        },
        "acrs_resnet34_vox2_random_gate_init.yaml": {
            "ablation.mode": "random_gate_init",
        },
        "acrs_resnet34_vox2_no_fusion.yaml": {
            "ablation.mode": "no_fusion_gate",
        },
    }

    for name, changes in expected.items():
        actual = load_config(name)
        expected_config = deepcopy(base)
        expected_config["exp_dir"] = actual["exp_dir"]
        args = expected_config["model_args"]["acrs_args"]
        for key, value in changes.items():
            target = args
            parts = key.split(".")
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = value
        assert actual == expected_config


def make_model(name):
    return ACRS(acrs_args=acrs_args(name)).eval()


@torch.no_grad()
def test_ablation_forward_paths():
    full = make_model("acrs_resnet34_vox2.yaml")
    state = full.state_dict()
    features = torch.linspace(-1.0, 1.0, 2 * 64 * 80).reshape(2, 64, 80)
    full_output = full(features)

    no_age_supervision = make_model(
        "acrs_resnet34_vox2_no_age_supervision.yaml")
    no_age_supervision.load_state_dict(state)
    no_age_supervision_output = no_age_supervision(features)
    torch.testing.assert_close(
        no_age_supervision_output["embedding"], full_output["embedding"])

    s3_only = make_model("acrs_resnet34_vox2_s3_only.yaml")
    s3_only.load_state_dict(state)
    s3_output = s3_only(features)
    assert torch.count_nonzero(s3_output["residual3_gate"])
    assert not torch.count_nonzero(s3_output["residual4_gate"])
    assert not s3_only.fusion.gate_off

    s4_only = make_model("acrs_resnet34_vox2_s4_only.yaml")
    s4_only.load_state_dict(state)
    s4_output = s4_only(features)
    assert not torch.count_nonzero(s4_output["residual3_gate"])
    assert torch.count_nonzero(s4_output["residual4_gate"])
    assert not s4_only.fusion.gate_off

    fusion_only = make_model("acrs_resnet34_vox2_fusion_only.yaml")
    fusion_only.load_state_dict(state)
    fusion_output = fusion_only(features)
    assert not torch.count_nonzero(fusion_output["residual3_gate"])
    assert not torch.count_nonzero(fusion_output["residual4_gate"])
    assert not fusion_only.fusion.gate_off

    no_conditioning = make_model(
        "acrs_resnet34_vox2_no_age_conditioning.yaml")
    no_conditioning.load_state_dict(state)
    no_conditioning_output = no_conditioning(features)
    assert not torch.count_nonzero(no_conditioning_output["residual3_gate"])
    assert not torch.count_nonzero(no_conditioning_output["residual4_gate"])
    assert no_conditioning.fusion.gate_off

    no_fusion = make_model("acrs_resnet34_vox2_no_fusion.yaml")
    no_fusion.load_state_dict(state)
    no_fusion_output = no_fusion(features)
    torch.testing.assert_close(
        no_fusion_output["residual3_gate"],
        full_output["residual3_gate"])
    torch.testing.assert_close(
        no_fusion_output["residual4_gate"],
        full_output["residual4_gate"])
    assert no_fusion.fusion.gate_off


def test_random_gate_initialization():
    model = make_model("acrs_resnet34_vox2_random_gate_init.yaml")
    torch.testing.assert_close(
        model.residual3.gate.bias,
        torch.zeros_like(model.residual3.gate.bias),
    )
    torch.testing.assert_close(
        model.residual4.gate.bias,
        torch.zeros_like(model.residual4.gate.bias),
    )
