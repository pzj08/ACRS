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


CONDITIONS = ("Correct", "Near-Age", "Shuffle-S3", "Shuffle-S4",
              "Shuffle-Both", "Far-Age")


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


def donor_indices(path, cache_index):
    mapping = {}
    with open(path) as stream:
        for row in csv.DictReader(stream):
            intervention = row["intervention_type"]
            if intervention in ("Shuffle-S3", "Near-Age", "Far-Age"):
                mapping[(row["target_utt"], intervention)] = cache_index[
                    row["donor_utt"]]
    return mapping


def counterfactual_embeddings(model, acts, target_g4, near_g3, near_g4,
                              shuffle_g3, shuffle_g4, far_g3, far_g4):
    i3 = acts["i3"]
    candidate3 = model.residual3.res_conv(i3)
    i3r_shuffle = i3 - shuffle_g3[:, :, None, None] * candidate3
    i3r_far = i3 - far_g3[:, :, None, None] * candidate3
    i3r_near = i3 - near_g3[:, :, None, None] * candidate3
    alternative_i4 = model.id_layer4(torch.cat(
        (i3r_shuffle, i3r_far), dim=0))
    shuffle_i4 = alternative_i4[0:1]
    far_i4 = alternative_i4[1:2]
    identity4 = torch.cat(
        (shuffle_i4, acts["i4"], shuffle_i4, far_i4), dim=0)
    gates4 = torch.cat(
        (target_g4, shuffle_g4, shuffle_g4, far_g4), dim=0)
    residual4 = model.residual4.res_conv(identity4)
    refined4 = identity4 - gates4[:, :, None, None] * residual4

    target_fusion = model.fusion_gate.low_rank(acts["a4"]).expand(
        refined4.shape[0], -1, -1, -1)
    fusion_gate = torch.sigmoid(model.fusion_gate.gate_conv(
        torch.cat((refined4, target_fusion), dim=1)))
    fused = refined4 * fusion_gate
    embedded = model.embedding_pool(fused)

    near_i4 = model.id_layer4(i3r_near)
    near_residual4 = model.residual4.res_conv(near_i4)
    near_refined4 = near_i4 - near_g4[:, :, None, None] * near_residual4
    near_fusion = model.fusion_gate(near_refined4, acts["a4"])
    near_embedded = model.embedding_pool(near_fusion)
    return {
        "Near-Age": near_embedded, "Shuffle-S3": embedded[0:1],
        "Shuffle-S4": embedded[1:2], "Shuffle-Both": embedded[2:3],
        "Far-Age": embedded[3:4],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--shard-list", required=True)
    parser.add_argument("--key-list", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--donor-mapping", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--sanity-count", type=int, default=16)
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cudnn.benchmark = False

    cache = torch.load(args.cache, map_location="cpu")
    cache_index = {utt: index for index, utt in enumerate(cache["utt_ids"])}
    mappings = donor_indices(args.donor_mapping, cache_index)
    model, config = load_model(args.config, args.checkpoint, device)
    dataset, test = make_dataset(config, args.shard_list, args.key_list)
    loader_args = dict(dataset=dataset, batch_size=1, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)
    if args.num_workers > 0:
        loader_args["prefetch_factor"] = 2
    loader = DataLoader(**loader_args)

    embeddings = {condition: [] for condition in CONDITIONS}
    utt_ids = []
    sanity = []
    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            utt = batch["key"][0]
            features = batch["feat"].float().to(device, non_blocking=True)
            if test.get("cmvn", True):
                features = apply_cmvn(features, **test.get("cmvn_args", {}))
            acts = model(features, return_layer_acts=True)
            self_index = cache_index[utt]
            shuffle_index = mappings[(utt, "Shuffle-S3")]
            near_index = mappings[(utt, "Near-Age")]
            far_index = mappings[(utt, "Far-Age")]

            target_g4 = cache["gate4"][self_index:self_index + 1].to(device)
            shuffle_g3 = cache["gate3"][shuffle_index:shuffle_index + 1].to(device)
            shuffle_g4 = cache["gate4"][shuffle_index:shuffle_index + 1].to(device)
            near_g3 = cache["gate3"][near_index:near_index + 1].to(device)
            near_g4 = cache["gate4"][near_index:near_index + 1].to(device)
            far_g3 = cache["gate3"][far_index:far_index + 1].to(device)
            far_g4 = cache["gate4"][far_index:far_index + 1].to(device)

            result = counterfactual_embeddings(
                model, acts, target_g4, near_g3, near_g4, shuffle_g3,
                shuffle_g4, far_g3, far_g4)
            result["Correct"] = acts["embedding"]
            utt_ids.append(utt)
            for condition in CONDITIONS:
                embeddings[condition].append(result[condition].cpu())

            if len(sanity) < args.sanity_count:
                self_intervention = model(
                    features, intervene_a3=acts["a3"],
                    intervene_a4=acts["a4"])["embedding"]
                first = result["Shuffle-Both"]
                repeated = counterfactual_embeddings(
                    model, acts, target_g4, near_g3, near_g4, shuffle_g3,
                    shuffle_g4, far_g3, far_g4)["Shuffle-Both"]
                diff = (acts["embedding"] - self_intervention).abs()
                repeat_diff = (first - repeated).abs()
                sanity.append({
                    "utt_id": utt,
                    "self_max_abs": float(diff.max()),
                    "self_mean_abs": float(diff.mean()),
                    "self_cosine": float(torch.nn.functional.cosine_similarity(
                        acts["embedding"], self_intervention).item()),
                    "repeat_max_abs": float(repeat_diff.max()),
                    "repeat_mean_abs": float(repeat_diff.mean()),
                })
            if step % 1000 == 0:
                print(f"rank={rank} utterances={step}", flush=True)

    payload = {
        "utt_ids": utt_ids,
        "embeddings": {key: torch.cat(value, dim=0)
                       for key, value in embeddings.items()},
        "sanity": sanity,
        "rank": rank,
        "world_size": world_size,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"interventions_rank{rank}.pt"
    torch.save(payload, path)
    print(f"rank={rank} saved={path} count={len(utt_ids)}", flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
