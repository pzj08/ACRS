from __future__ import annotations

import math
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


AGE_MEAN = 35.0
AGE_STD = 15.0


def _merged_config(config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "age_bins": [18, 25, 35, 45, 55, 65],
        "ignore_age_index": -1,
        "num_posterior_bins": 4,
        "losses": {
            "lambda_age": 0.05,
            "lambda_consistency": 0.02,
            "lambda_path": 0.0,
            "ramp_epoch": 2.0,
            "training_counterfactual": False,
            "consistency_schedule": None,
        },
        "ablation": {"mode": "none"},
    }
    config = dict(config or {})
    result.update({k: v for k, v in config.items()
                   if k not in ("losses", "ablation")})
    result["losses"].update(dict(config.get("losses", {})))
    result["ablation"].update(dict(config.get("ablation", {})))

    bins = [float(value) for value in result["age_bins"]]
    if not bins or bins != sorted(bins) or len(set(bins)) != len(bins):
        raise ValueError(
            "age_bins must be a non-empty, strictly increasing list")
    result["age_bins"] = bins
    result["num_age_groups"] = len(bins) + 1
    if int(result["num_posterior_bins"]) < 2:
        raise ValueError("num_posterior_bins must be at least 2")

    mode = str(result["ablation"]["mode"])
    valid_modes = {
        "none",
        "no_age_conditioning",
        "no_residual_suppression",
        "no_fusion_gate",
        "random_gate_init",
    }
    if mode not in valid_modes:
        raise ValueError(
            f"unknown ablation mode {mode!r}; expected one of "
            f"{sorted(valid_modes)}")
    return result


def _statistics(x: torch.Tensor) -> torch.Tensor:
    x = x.flatten(2)
    mean = x.mean(dim=2)
    variance = x.var(dim=2, unbiased=False)
    return torch.cat([mean, torch.sqrt(variance + 1.0e-12)], dim=1)


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, channels, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        if stride != 1 or in_channels != channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


def _make_layer(in_channels: int, channels: int, blocks: int,
                stride: int) -> nn.Sequential:
    layers = [BasicBlock(in_channels, channels, stride)]
    layers.extend(BasicBlock(channels, channels) for _ in range(1, blocks))
    return nn.Sequential(*layers)


class AgeConditionedResidualBlock(nn.Module):
    def __init__(self, channels: int, *, gate_init: str = "default",
                 disabled: bool = False):
        super().__init__()
        self.disabled = disabled
        self.residual = nn.Conv2d(channels, channels, kernel_size=1)
        self.age_projection = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
        )
        self.gate = nn.Linear(channels, channels)
        if gate_init == "random":
            nn.init.kaiming_normal_(self.gate.weight, a=math.sqrt(5))
            nn.init.zeros_(self.gate.bias)
        else:
            nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, identity: torch.Tensor,
                age: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.disabled:
            gate = identity.new_zeros((identity.shape[0], identity.shape[1]))
            return identity, gate
        age_code = self.age_projection(_statistics(age))
        gate = torch.sigmoid(self.gate(age_code))
        residual = self.residual(identity)
        output = identity - gate[:, :, None, None] * residual
        return output, gate


class AgeConditionedFusionGate(nn.Module):
    def __init__(self, channels: int, low_rank: int = 32,
                 disabled: bool = False):
        super().__init__()
        self.disabled = disabled
        self.age_projection = nn.Conv2d(
            channels, low_rank, kernel_size=1, bias=False)
        self.gate = nn.Conv2d(channels + low_rank, channels, kernel_size=1)

    def forward(self, identity: torch.Tensor,
                age: torch.Tensor) -> torch.Tensor:
        if self.disabled:
            return identity
        age = self.age_projection(age)
        gate = torch.sigmoid(self.gate(torch.cat([identity, age], dim=1)))
        return identity * gate


class AttentiveSpatiotemporalStatisticsPooling(nn.Module):
    def __init__(self, channels: int, embedding_dim: int,
                 eps: float = 1.0e-12):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.attention = nn.Conv2d(channels, 1, kernel_size=1)
        self.embedding = nn.Linear(2 * channels, embedding_dim)
        self.normalization = nn.BatchNorm1d(embedding_dim)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        batch, channels, frequency, time = x.shape
        weights = self.attention(x).reshape(batch, frequency * time)
        weights = torch.softmax(weights, dim=1).unsqueeze(1)
        x = x.reshape(batch, channels, frequency * time)
        mean = torch.sum(weights * x, dim=2)
        second_moment = torch.sum(weights * x.square(), dim=2)
        std = torch.sqrt(torch.clamp(second_moment - mean.square(), min=0.0)
                         + self.eps)
        embedding = self.embedding(torch.cat([mean, std], dim=1))
        embedding = self.normalization(embedding)
        return F.normalize(embedding, p=2, dim=1)


class AgeHead(nn.Module):
    def __init__(self, channels: int, posterior_bins: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(2 * channels, 128),
            nn.ReLU(inplace=True),
        )
        self.regression = nn.Linear(128, 1)
        self.classification = nn.Linear(128, posterior_bins)

    def forward(self, age: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared(_statistics(age))
        return self.regression(hidden).squeeze(-1), self.classification(hidden)


class ACRS(nn.Module):
    def __init__(self, feat_dim: int = 80, embed_dim: int = 256,
                 acrs_args: Optional[Mapping[str, Any]] = None,
                 **_: Any):
        super().__init__()
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.config = _merged_config(acrs_args)
        self.num_age_groups = int(self.config["num_age_groups"])
        self.ignore_age_index = int(self.config["ignore_age_index"])

        losses = self.config["losses"]
        self.age_loss_weight = float(losses["lambda_age"])
        self.consistency_weight = float(losses["lambda_consistency"])
        self.consistency_schedule = losses["consistency_schedule"]
        self.path_weight = float(losses["lambda_path"])
        self.ramp_epoch = float(losses["ramp_epoch"])
        self.use_training_counterfactual = bool(
            losses["training_counterfactual"])

        centers = self._age_bin_centers(self.config["age_bins"])
        self.register_buffer(
            "age_bin_centers", torch.tensor(centers, dtype=torch.float32))

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = _make_layer(32, 32, 3, 1)
        self.layer2 = _make_layer(32, 64, 4, 2)

        self.age_layer3 = _make_layer(64, 128, 6, 2)
        self.age_layer4 = _make_layer(128, 256, 3, 2)
        self.age_head = AgeHead(256, int(self.config["num_posterior_bins"]))

        mode = self.config["ablation"]["mode"]
        residual_disabled = mode in {
            "no_age_conditioning", "no_residual_suppression"}
        fusion_disabled = mode in {
            "no_age_conditioning", "no_fusion_gate"}
        gate_init = "random" if mode == "random_gate_init" else "default"

        self.identity_layer3 = _make_layer(64, 128, 6, 2)
        self.residual3 = AgeConditionedResidualBlock(
            128, gate_init=gate_init, disabled=residual_disabled)
        self.identity_layer4 = _make_layer(128, 256, 3, 2)
        self.residual4 = AgeConditionedResidualBlock(
            256, gate_init=gate_init, disabled=residual_disabled)
        self.fusion = AgeConditionedFusionGate(
            256, disabled=fusion_disabled)
        self.pooling = AttentiveSpatiotemporalStatisticsPooling(
            256, embed_dim)
        self.embedding_age_readout = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    @staticmethod
    def _age_bin_centers(boundaries: list[float]) -> list[float]:
        first = boundaries[0] - (boundaries[1] - boundaries[0]) / 2.0 \
            if len(boundaries) > 1 else boundaries[0]
        last = boundaries[-1] + (boundaries[-1] - boundaries[-2]) / 2.0 \
            if len(boundaries) > 1 else boundaries[0]
        middle = [(left + right) / 2.0
                  for left, right in zip(boundaries[:-1], boundaries[1:])]
        return [first, *middle, last]

    def _identity_path(
        self, shared: torch.Tensor, age3: torch.Tensor, age4: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        identity3 = self.identity_layer3(shared)
        identity3, gate3 = self.residual3(identity3, age3)
        identity4 = self.identity_layer4(identity3)
        identity4, gate4 = self.residual4(identity4, age4)
        identity4 = self.fusion(identity4, age4)
        embedding = self.pooling(identity4)
        return identity3, gate3, identity4, gate4, embedding

    def _counterfactual_permutation(self,
                                    age_groups: torch.Tensor) -> torch.Tensor:
        indices = torch.arange(age_groups.shape[0], device=age_groups.device)
        valid = age_groups != self.ignore_age_index
        permutation = indices.clone()
        if valid.sum() > 1:
            valid_indices = indices[valid]
            order = torch.argsort(age_groups[valid], stable=True)
            sorted_indices = valid_indices[order]
            offset = sorted_indices.shape[0] // 2
            permutation[valid_indices] = torch.cat(
                [sorted_indices[offset:], sorted_indices[:offset]])
        return permutation

    def forward(
        self,
        features: torch.Tensor,
        *,
        age_groups: Optional[torch.Tensor] = None,
        intervene_age3: Optional[torch.Tensor] = None,
        intervene_age4: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Dict[str, Any]:
        if features.ndim != 3 or features.shape[2] != self.feat_dim:
            raise ValueError(
                f"expected features [B, T, {self.feat_dim}], got "
                f"{tuple(features.shape)}")

        x = features.permute(0, 2, 1).unsqueeze(1)
        stem = F.relu(self.bn1(self.conv1(x)), inplace=True)
        shared1 = self.layer1(stem)
        shared2 = self.layer2(shared1)

        age3 = self.age_layer3(shared2)
        age4 = self.age_layer4(age3)
        age_prediction, age_logits = self.age_head(age4)
        conditioned_age3 = age3 if intervene_age3 is None else intervene_age3
        conditioned_age4 = age4 if intervene_age4 is None else intervene_age4
        identity3, gate3, identity4, gate4, embedding = self._identity_path(
            shared2, conditioned_age3, conditioned_age4)

        output: Dict[str, Any] = {
            "embedding": embedding,
            "age_prediction": age_prediction,
            "age_logits": age_logits,
            "residual3_gate": gate3,
            "residual4_gate": gate4,
        }

        counterfactual_active = (
            self.use_training_counterfactual
            and age_groups is not None
            and intervene_age3 is None
            and intervene_age4 is None
        )
        if counterfactual_active:
            permutation = self._counterfactual_permutation(age_groups)
            *_, counterfactual_embedding = self._identity_path(
                shared2, age3[permutation], age4[permutation])
        else:
            permutation = torch.arange(
                embedding.shape[0], device=embedding.device)
            counterfactual_embedding = embedding
        output.update({
            "counterfactual_embedding": counterfactual_embedding,
            "counterfactual_permutation": permutation,
            "counterfactual_active": counterfactual_active,
        })

        if return_features:
            output["features"] = {
                "shared1": shared1,
                "shared2": shared2,
                "age3": age3,
                "age4": age4,
                "identity3": identity3,
                "identity4": identity4,
            }
        return output

    def compute_acrs_losses(
        self,
        output: Mapping[str, Any],
        age_groups: torch.Tensor,
        *,
        speakers: Optional[torch.Tensor] = None,
        speaker_classifier: Optional[Callable[..., torch.Tensor]] = None,
        epoch: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        device = age_groups.device
        zero = torch.zeros((), device=device)
        valid = age_groups != self.ignore_age_index

        gate3 = output["residual3_gate"]
        gate4 = output["residual4_gate"]
        retention = torch.cat([(1.0 - gate3).flatten(),
                               (1.0 - gate4).flatten()])
        probabilities = F.softmax(output["age_logits"], dim=1)
        entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()) \
            .sum(dim=1).mean()

        losses = {
            "loss_age": zero.clone(),
            "loss_consistency": zero.clone(),
            "loss_path": zero.clone(),
            "loss_acrs_total": zero.clone(),
            "gate_retention_mean": retention.mean(),
            "gate_retention_std": retention.std(),
            "age_posterior_entropy": entropy,
        }
        if not valid.any():
            return losses

        groups = age_groups[valid]
        if (groups < 0).any() or (groups >= self.num_age_groups).any():
            raise ValueError(
                f"valid age groups must be in [0, {self.num_age_groups - 1}]")
        targets = (self.age_bin_centers[groups] - AGE_MEAN) / AGE_STD
        continuous_loss = F.mse_loss(output["age_prediction"][valid], targets)
        posterior_bins = int(self.config["num_posterior_bins"])
        posterior_targets = torch.clamp(
            groups * posterior_bins // self.num_age_groups,
            max=posterior_bins - 1,
        )
        classification_loss = F.cross_entropy(
            output["age_logits"][valid], posterior_targets)
        losses["loss_age"] = continuous_loss + classification_loss

        if output["counterfactual_active"]:
            embedding = output["embedding"]
            counterfactual = output["counterfactual_embedding"]
            permutation = output["counterfactual_permutation"]
            counterfactual_targets = (
                self.age_bin_centers[age_groups[permutation[valid]]]
                - AGE_MEAN
            ) / AGE_STD
            real_readout = self.embedding_age_readout(embedding).squeeze(-1)
            counterfactual_readout = self.embedding_age_readout(
                counterfactual).squeeze(-1)
            losses["loss_path"] = (
                F.mse_loss(real_readout[valid], targets)
                + F.mse_loss(counterfactual_readout[valid],
                             counterfactual_targets)
            )
            if speaker_classifier is not None:
                if speakers is None:
                    raise ValueError(
                        "speakers are required when speaker_classifier is set")
                try:
                    logits = speaker_classifier(counterfactual, speakers)
                except TypeError:
                    logits = speaker_classifier(counterfactual)
                losses["loss_consistency"] = F.cross_entropy(logits, speakers)

        consistency_weight = self.consistency_weight
        if self.consistency_schedule is not None and epoch is not None:
            schedule = self.consistency_schedule
            start_epoch = float(schedule["start_epoch"])
            end_epoch = float(schedule["end_epoch"])
            if end_epoch <= start_epoch:
                raise ValueError(
                    "consistency_schedule.end_epoch must exceed start_epoch")
            progress = min(1.0, max(
                0.0, (float(epoch) - start_epoch) / (end_epoch - start_epoch)))
            consistency_weight = (
                float(schedule["start_weight"])
                + progress * (float(schedule["end_weight"])
                              - float(schedule["start_weight"]))
            )

        ramp = 1.0 if epoch is None or self.ramp_epoch <= 0 else min(
            1.0, max(0.0, float(epoch) / self.ramp_epoch))
        losses["loss_acrs_total"] = ramp * (
            self.age_loss_weight * losses["loss_age"]
            + consistency_weight * losses["loss_consistency"]
            + self.path_weight * losses["loss_path"]
        )
        return losses


def ACRS_ResNet34(
    feat_dim: int = 80,
    embed_dim: int = 256,
    pooling_func: str = "ACRS",
    two_emb_layer: bool = True,
    acrs_args: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> ACRS:
    del pooling_func, two_emb_layer
    return ACRS(feat_dim=feat_dim, embed_dim=embed_dim,
                acrs_args=acrs_args, **kwargs)
