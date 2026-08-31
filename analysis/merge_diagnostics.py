import argparse
import csv
import hashlib
import json
from pathlib import Path

import kaldiio
import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", required=True)
    parser.add_argument("--reference-scp", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--verification-output", required=True)
    args = parser.parse_args()

    parts = [torch.load(path, map_location="cpu") for path in
             sorted(Path(args.input_dir).glob("diagnostics_rank*.pt"))]
    if not parts:
        raise RuntimeError("no diagnostic shards found")
    list_keys = ("utt_ids", "speaker_ids")
    excluded = set(list_keys) | {"rank", "world_size"}
    combined = {}
    for key in list_keys:
        combined[key] = sum((part[key] for part in parts), [])
    for key in parts[0]:
        if key not in excluded:
            combined[key] = torch.cat([part[key] for part in parts], dim=0)

    with open(args.manifest) as stream:
        manifest_rows = list(csv.DictReader(stream))
    expected = [row["utt_id"] for row in manifest_rows]
    source_index = {utt: index for index, utt in enumerate(combined["utt_ids"])}
    missing = sorted(set(expected) - set(source_index))
    extra = sorted(set(source_index) - set(expected))
    duplicate_count = len(combined["utt_ids"]) - len(source_index)
    if missing or extra or duplicate_count:
        raise RuntimeError(
            f"cache mismatch missing={len(missing)} extra={len(extra)} "
            f"duplicates={duplicate_count}")
    order = torch.tensor([source_index[utt] for utt in expected])
    for key in list_keys:
        combined[key] = [combined[key][index] for index in order.tolist()]
    for key in list(combined):
        if key not in list_keys:
            combined[key] = combined[key][order]
    combined["checkpoint"] = str(Path(args.checkpoint).resolve())
    combined["checkpoint_sha256"] = sha256(args.checkpoint)
    combined["evaluation_mode"] = "whole_utterance"
    combined["age_boundaries"] = [21.0, 31.0, 41.0, 51.0, 61.0, 71.0]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(combined, output)

    gate3 = combined["gate3"].numpy()
    gate4 = combined["gate4"].numpy()
    fields = [
        "utt_id", "speaker_id", "age", "age_group", "num_frames",
        "gate3_mean", "gate3_std", "gate4_mean", "gate4_std",
        "suppression_ratio_stage3", "suppression_ratio_stage4",
        "age_prediction_years",
    ]
    with open(args.metadata_output, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, utt in enumerate(expected):
            writer.writerow({
                "utt_id": utt,
                "speaker_id": combined["speaker_ids"][index],
                "age": f"{combined['ages'][index].item():.6f}",
                "age_group": combined["age_groups"][index].item(),
                "num_frames": combined["num_frames"][index].item(),
                "gate3_mean": f"{gate3[index].mean():.9f}",
                "gate3_std": f"{gate3[index].std():.9f}",
                "gate4_mean": f"{gate4[index].mean():.9f}",
                "gate4_std": f"{gate4[index].std():.9f}",
                "suppression_ratio_stage3":
                    f"{combined['suppression_ratio_stage3'][index].item():.9f}",
                "suppression_ratio_stage4":
                    f"{combined['suppression_ratio_stage4'][index].item():.9f}",
                "age_prediction_years":
                    f"{combined['age_predictions_years'][index].item():.6f}",
            })

    cached = {utt: combined["embeddings"][index].numpy()
              for index, utt in enumerate(expected)}
    compared = 0
    max_abs = 0.0
    sum_abs = 0.0
    count = 0
    min_cosine = 1.0
    for utt, reference in kaldiio.load_scp_sequential(args.reference_scp):
        if utt not in cached:
            continue
        difference = np.abs(cached[utt] - reference)
        max_abs = max(max_abs, float(difference.max()))
        sum_abs += float(difference.sum())
        count += difference.size
        cosine = float(np.dot(cached[utt], reference) /
                       (np.linalg.norm(cached[utt]) * np.linalg.norm(reference)))
        min_cosine = min(min_cosine, cosine)
        compared += 1
    verification = {
        "expected_utterances": len(expected),
        "compared_with_existing_embeddings": compared,
        "max_absolute_difference": max_abs,
        "mean_absolute_difference": sum_abs / max(count, 1),
        "minimum_cosine_similarity": min_cosine,
        "checkpoint_sha256": combined["checkpoint_sha256"],
    }
    with open(args.verification_output, "w") as stream:
        json.dump(verification, stream, indent=2)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
