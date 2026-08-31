import argparse
import copy
import csv
import os
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader

from wespeaker.dataset.dataset import Dataset
from wespeaker.dataset.dataset_utils import apply_cmvn
from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.checkpoint import load_checkpoint


def statistics(features):
    flat = features.flatten(2)
    mean = flat.mean(dim=2)
    variance = flat.var(dim=2, unbiased=False)
    return torch.cat((mean, torch.sqrt(variance + 1.0e-12)), dim=1)


def load_manifest(path):
    metadata = {}
    with open(path) as stream:
        for row in csv.DictReader(stream):
            metadata[row["utt_id"]] = (
                row["speaker_id"], float(row["age"]), int(row["age_group"]))
    return metadata


def load_model(config_path, checkpoint, device):
    with open(config_path) as stream:
        config = yaml.safe_load(stream)
    config["model"] = "ACRS"
    config["model_args"]["pooling_func"] = "ACRS"
    model = get_speaker_model("ACRS")(**config["model_args"])
    report = load_checkpoint(model, checkpoint)
    missing = [key for key in report["missing_keys"] if "projection" not in key]
    unexpected = [key for key in report["unexpected_keys"]
                  if "projection" not in key]
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    return model.to(device).eval(), config


def make_dataset(config, shard_list, key_list):
    test = copy.deepcopy(config["dataset_args"])
    test["speed_perturb"] = False
    test["fbank_args"]["dither"] = 0.0
    test["spec_aug"] = False
    test["shuffle"] = False
    test["aug_prob"] = 0.0
    test["filter"] = False
    dataset = Dataset(
        "shard", shard_list, test, {}, whole_utt=True,
        repeat_dataset=False, key_filter_file=key_list)
    return dataset, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--shard-list", required=True)
    parser.add_argument("--key-list", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cudnn.benchmark = False

    metadata = load_manifest(args.manifest)
    model, config = load_model(args.config, args.checkpoint, device)
    dataset, test = make_dataset(config, args.shard_list, args.key_list)
    loader_args = dict(
        dataset=dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    if args.num_workers > 0:
        loader_args["prefetch_factor"] = 2
    loader = DataLoader(**loader_args)

    values = {
        "utt_ids": [],
        "speaker_ids": [],
        "ages": [],
        "age_groups": [],
        "num_frames": [],
        "embeddings": [],
        "gate3": [],
        "gate4": [],
        "age_predictions_normalized": [],
        "age_predictions_years": [],
        "age_logits": [],
        "age3_statistics": [],
        "age4_statistics": [],
        "fusion_condition": [],
        "suppression_ratio_stage3": [],
        "suppression_ratio_stage4": [],
    }

    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            utt = batch["key"][0]
            speaker, age, group = metadata[utt]
            features = batch["feat"].float().to(device, non_blocking=True)
            if test.get("cmvn", True):
                features = apply_cmvn(features,
                                      **test.get("cmvn_args", {}))
            output = model(features, return_layer_acts=True)
            candidate3 = model.residual3.res_conv(output["i3"])
            candidate4 = model.residual4.res_conv(output["i4"])
            suppressed3 = output["res3_gate"][:, :, None, None] * candidate3
            suppressed4 = output["res4_gate"][:, :, None, None] * candidate4
            ratio3 = (suppressed3.flatten(1).norm(dim=1)
                      / output["i3"].flatten(1).norm(dim=1).clamp_min(1.0e-12))
            ratio4 = (suppressed4.flatten(1).norm(dim=1)
                      / output["i4"].flatten(1).norm(dim=1).clamp_min(1.0e-12))
            fusion = model.fusion_gate.low_rank(output["a4"]).mean(dim=3)

            values["utt_ids"].append(utt)
            values["speaker_ids"].append(speaker)
            values["ages"].append(age)
            values["age_groups"].append(group)
            values["num_frames"].append(int(features.shape[1]))
            values["embeddings"].append(output["embedding"].cpu())
            values["gate3"].append(output["res3_gate"].cpu())
            values["gate4"].append(output["res4_gate"].cpu())
            prediction = output["age_pred"].cpu()
            values["age_predictions_normalized"].append(prediction)
            values["age_predictions_years"].append(prediction * 15.0 + 35.0)
            values["age_logits"].append(output["age_q_logits"].cpu())
            values["age3_statistics"].append(statistics(output["a3"]).cpu())
            values["age4_statistics"].append(statistics(output["a4"]).cpu())
            values["fusion_condition"].append(fusion.cpu())
            values["suppression_ratio_stage3"].append(ratio3.cpu())
            values["suppression_ratio_stage4"].append(ratio4.cpu())
            if step % 1000 == 0:
                print(f"rank={rank} utterances={step}", flush=True)

    tensor_keys = [key for key in values
                   if key not in ("utt_ids", "speaker_ids", "ages",
                                  "age_groups", "num_frames")]
    for key in tensor_keys:
        values[key] = torch.cat(values[key], dim=0)
    values["ages"] = torch.tensor(values["ages"], dtype=torch.float64)
    values["age_groups"] = torch.tensor(values["age_groups"], dtype=torch.int64)
    values["num_frames"] = torch.tensor(values["num_frames"], dtype=torch.int32)
    values["rank"] = rank
    values["world_size"] = world_size

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"diagnostics_rank{rank}.pt"
    torch.save(values, path)
    print(f"rank={rank} saved={path} count={len(values['utt_ids'])}", flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
