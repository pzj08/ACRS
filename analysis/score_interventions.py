import argparse
import csv
import json
from pathlib import Path

import kaldiio
import numpy as np
import torch
from sklearn.preprocessing import normalize

from wespeaker.utils.score_metrics import (compute_c_norm, compute_eer,
                                           compute_pmiss_pfa_rbst)


CONDITIONS = ("Correct", "Near-Age", "Shuffle-S3", "Shuffle-S4",
              "Shuffle-Both", "Far-Age")
PROTOCOL = "cosine_without_mean_subtraction"


def load_trials(paths):
    datasets = {}
    for path in paths:
        rows = []
        with open(path) as stream:
            for line in stream:
                first, second, label = line.split()
                rows.append((first, second, label))
        datasets[Path(path).stem] = rows
    return datasets


def score_metrics(scores, labels):
    fnr, fpr = compute_pmiss_pfa_rbst(scores, labels)
    eer, _ = compute_eer(fnr, fpr, scores)
    dcf = compute_c_norm(fnr, fpr, p_target=0.01, c_miss=1, c_fa=1)
    return float(100.0 * eer), float(dcf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--correct-reference-scp", required=True)
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--skip-embedding-write", action="store_true")
    parser.add_argument("--score-dir", required=True)
    parser.add_argument("--results-csv", required=True)
    parser.add_argument("--results-md", required=True)
    parser.add_argument("--near-far-results-csv", required=True)
    parser.add_argument("--near-far-results-md", required=True)
    parser.add_argument("--sanity-md", required=True)
    parser.add_argument("--verification-json", required=True)
    parser.add_argument("trials", nargs="+")
    args = parser.parse_args()

    parts = [torch.load(path, map_location="cpu") for path in
             sorted(Path(args.input_dir).glob("interventions_rank*.pt"))]
    if not parts:
        raise RuntimeError("no intervention shards found")
    utterances = sum((part["utt_ids"] for part in parts), [])
    combined = {condition: torch.cat(
        [part["embeddings"][condition] for part in parts], dim=0)
        for condition in CONDITIONS}
    source_index = {utt: index for index, utt in enumerate(utterances)}
    with open(args.manifest) as stream:
        expected = [row["utt_id"] for row in csv.DictReader(stream)]
    if len(source_index) != len(utterances) or set(source_index) != set(expected):
        raise RuntimeError("intervention cache does not match manifest")
    order = torch.tensor([source_index[utt] for utt in expected])
    for condition in CONDITIONS:
        combined[condition] = combined[condition][order]
    index = {utt: position for position, utt in enumerate(expected)}

    embedding_dir = Path(args.embedding_dir)
    embedding_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_embedding_write:
        for condition in CONDITIONS:
            stem = condition.lower().replace("-", "_")
            ark = embedding_dir / f"{stem}.ark"
            scp = embedding_dir / f"{stem}.scp"
            with kaldiio.WriteHelper(f"ark,scp:{ark},{scp}") as writer:
                matrix = combined[condition].numpy()
                for row_index, utt in enumerate(expected):
                    writer(utt, matrix[row_index])

    correct_reference_max_abs = 0.0
    correct_reference_mean_sum = 0.0
    correct_reference_count = 0
    correct_values = combined["Correct"].numpy()
    for utt, reference in kaldiio.load_scp_sequential(args.correct_reference_scp):
        if utt not in index:
            continue
        difference = np.abs(correct_values[index[utt]] - reference)
        correct_reference_max_abs = max(correct_reference_max_abs,
                                        float(difference.max()))
        correct_reference_mean_sum += float(difference.sum())
        correct_reference_count += difference.size

    arrays = {}
    for condition in CONDITIONS:
        values = combined[condition].numpy()
        arrays[condition] = normalize(values, copy=True)
    datasets = load_trials(args.trials)
    score_dir = Path(args.score_dir)
    score_dir.mkdir(parents=True, exist_ok=True)
    result_rows = []
    correct_by_dataset = {}
    near_by_dataset = {}
    for dataset, rows in datasets.items():
        left = np.fromiter((index[row[0]] for row in rows), dtype=np.int64)
        right = np.fromiter((index[row[1]] for row in rows), dtype=np.int64)
        labels = np.fromiter((row[2] == "target" for row in rows), dtype=bool)
        for condition in CONDITIONS:
            values = arrays[condition]
            scores = np.matmul(values[left, None, :],
                               values[right, :, None])[:, 0, 0]
            path = score_dir / f"{dataset}.{condition.lower().replace('-', '_')}.score"
            quantized = np.empty(len(scores), dtype=np.float64)
            with open(path, "w") as stream:
                for number, (row, score) in enumerate(zip(rows, scores)):
                    rendered = f"{score:.5f}"
                    quantized[number] = float(rendered)
                    stream.write(f"{row[0]} {row[1]} {rendered} {row[2]}\n")
            eer, dcf = score_metrics(quantized, labels)
            if condition == "Correct":
                correct_by_dataset[dataset] = eer
            elif condition == "Near-Age":
                near_by_dataset[dataset] = (eer, dcf)
            result_rows.append({"Protocol": PROTOCOL,
                                "Dataset": dataset, "Condition": condition,
                                "EER": eer, "minDCF": dcf})
            print(dataset, condition, f"EER={eer:.3f}", f"minDCF={dcf:.3f}",
                  flush=True)

    for row in result_rows:
        baseline = correct_by_dataset[row["Dataset"]]
        absolute = row["EER"] - baseline
        relative = 100.0 * absolute / baseline if baseline else float("nan")
        row["absolute_EER_degradation"] = absolute
        row["relative_EER_degradation_percent"] = relative
        if row["Condition"] == "Far-Age":
            near_eer, near_dcf = near_by_dataset[row["Dataset"]]
            near_far_eer = row["EER"] - near_eer
            row["Near_to_Far_absolute_EER_degradation"] = near_far_eer
            row["Near_to_Far_relative_EER_degradation_percent"] = (
                100.0 * near_far_eer / near_eer)
            row["Near_to_Far_minDCF_degradation"] = row["minDCF"] - near_dcf
        else:
            row["Near_to_Far_absolute_EER_degradation"] = ""
            row["Near_to_Far_relative_EER_degradation_percent"] = ""
            row["Near_to_Far_minDCF_degradation"] = ""
    with open(args.results_csv, "w", newline="") as stream:
        fields = list(result_rows[0])
        writer = csv.DictWriter(stream, fieldnames=fields,
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(result_rows)

    with open(args.results_md, "w") as stream:
        stream.write("# Counterfactual age-conditioning results\n\n")
        stream.write("All values use cosine scoring without mean "
                     "subtraction. Cells report EER (%) / minDCF.\n\n")
        stream.write("| Dataset | " + " | ".join(CONDITIONS) + " |\n")
        stream.write("|---" * (len(CONDITIONS) + 1) + "|\n")
        for dataset in datasets:
            cells = []
            for condition in CONDITIONS:
                row = next(item for item in result_rows
                           if item["Dataset"] == dataset and
                           item["Condition"] == condition)
                cells.append(f"{row['EER']:.3f} / {row['minDCF']:.3f}")
            stream.write(f"| {dataset} | " + " | ".join(cells) + " |\n")

    near_far_rows = []
    for dataset in datasets:
        correct = next(row for row in result_rows
                       if row["Dataset"] == dataset and
                       row["Condition"] == "Correct")
        near = next(row for row in result_rows
                    if row["Dataset"] == dataset and
                    row["Condition"] == "Near-Age")
        far = next(row for row in result_rows
                   if row["Dataset"] == dataset and
                   row["Condition"] == "Far-Age")
        near_far_rows.append({
            "Protocol": PROTOCOL,
            "Dataset": dataset,
            "Correct_EER": correct["EER"], "Correct_minDCF": correct["minDCF"],
            "Near_Age_EER": near["EER"], "Near_Age_minDCF": near["minDCF"],
            "Far_Age_EER": far["EER"], "Far_Age_minDCF": far["minDCF"],
            "Near_to_Far_absolute_EER_degradation":
                far["EER"] - near["EER"],
            "Near_to_Far_relative_EER_degradation_percent":
                100.0 * (far["EER"] - near["EER"]) / near["EER"],
            "Near_to_Far_minDCF_degradation": far["minDCF"] - near["minDCF"],
            "Far_EER_worse_than_Near": far["EER"] > near["EER"],
            "Far_minDCF_worse_than_Near": far["minDCF"] > near["minDCF"],
        })
    with open(args.near_far_results_csv, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(near_far_rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(near_far_rows)
    with open(args.near_far_results_md, "w") as stream:
        stream.write("# Near-Age versus Far-Age intervention\n\n")
        stream.write("All values use cosine scoring without mean "
                     "subtraction. EER is reported in percent.\n\n")
        stream.write("| Dataset | Correct EER/minDCF | Near-Age EER/minDCF | "
                     "Far-Age EER/minDCF | Near→Far ΔEER | Near→Far ΔminDCF |\n")
        stream.write("|---|---:|---:|---:|---:|---:|\n")
        for row in near_far_rows:
            stream.write(
                f"| {row['Dataset']} | {row['Correct_EER']:.3f} / "
                f"{row['Correct_minDCF']:.3f} | {row['Near_Age_EER']:.3f} / "
                f"{row['Near_Age_minDCF']:.3f} | {row['Far_Age_EER']:.3f} / "
                f"{row['Far_Age_minDCF']:.3f} | "
                f"{row['Near_to_Far_absolute_EER_degradation']:+.3f} | "
                f"{row['Near_to_Far_minDCF_degradation']:+.3f} |\n")
        all_eer_worse = all(row["Far_EER_worse_than_Near"]
                            for row in near_far_rows)
        all_dcf_worse = all(row["Far_minDCF_worse_than_Near"]
                            for row in near_far_rows)
        eer_changes = [row["Near_to_Far_absolute_EER_degradation"]
                       for row in near_far_rows]
        dcf_changes = [row["Near_to_Far_minDCF_degradation"]
                       for row in near_far_rows]
        stream.write(
            "\nFar-Age is worse than Near-Age on "
            f"{sum(row['Far_EER_worse_than_Near'] for row in near_far_rows)}/"
            f"{len(near_far_rows)} sets by EER and "
            f"{sum(row['Far_minDCF_worse_than_Near'] for row in near_far_rows)}/"
            f"{len(near_far_rows)} sets by minDCF. Near→Far increases EER by "
            f"{min(eer_changes):.3f}–{max(eer_changes):.3f} points and minDCF "
            f"by {min(dcf_changes):.3f}–{max(dcf_changes):.3f}. "
            f"Consistent ordering: {all_eer_worse and all_dcf_worse}.\n")

    sanity_rows = sum((part["sanity"] for part in parts), [])
    norm_summary = {}
    for condition in CONDITIONS:
        values = torch.linalg.vector_norm(combined[condition], dim=1).numpy()
        norm_summary[condition] = {
            "mean": float(values.mean()), "std": float(values.std()),
            "min": float(values.min()), "max": float(values.max())}
    self_max = max(row["self_max_abs"] for row in sanity_rows)
    self_mean = float(np.mean([row["self_mean_abs"] for row in sanity_rows]))
    self_cos = min(row["self_cosine"] for row in sanity_rows)
    repeat_max = max(row["repeat_max_abs"] for row in sanity_rows)
    repeat_mean = float(np.mean([row["repeat_mean_abs"] for row in sanity_rows]))
    with open(args.sanity_md, "w") as stream:
        stream.write("# Intervention sanity checks\n\n")
        stream.write(f"Self intervention ({len(sanity_rows)} utterances): "
                     f"max absolute difference {self_max:.9g}, mean absolute "
                     f"difference {self_mean:.9g}, minimum cosine similarity "
                     f"{self_cos:.9g}.\n\n")
        stream.write(f"Repeated fixed-donor intervention: max absolute "
                     f"difference {repeat_max:.9g}, mean absolute difference "
                     f"{repeat_mean:.9g}.\n\n")
        stream.write(f"Correct re-extraction versus the archived full-system "
                     f"embedding: max absolute difference "
                     f"{correct_reference_max_abs:.9g}, mean absolute "
                     f"difference {correct_reference_mean_sum / max(correct_reference_count, 1):.9g}.\n\n")
        stream.write("| Condition | Mean | Std | Min | Max |\n|---|---:|---:|---:|---:|\n")
        for condition, summary in norm_summary.items():
            stream.write(f"| {condition} | {summary['mean']:.9f} | "
                         f"{summary['std']:.9f} | {summary['min']:.9f} | "
                         f"{summary['max']:.9f} |\n")

    verification = {
        "protocol": PROTOCOL,
        "utterances": len(expected),
        "self_intervention_max_absolute_difference": self_max,
        "self_intervention_mean_absolute_difference": self_mean,
        "self_intervention_minimum_cosine_similarity": self_cos,
        "repeat_intervention_max_absolute_difference": repeat_max,
        "repeat_intervention_mean_absolute_difference": repeat_mean,
        "correct_vs_existing_max_absolute_difference":
            correct_reference_max_abs,
        "correct_vs_existing_mean_absolute_difference":
            correct_reference_mean_sum / max(correct_reference_count, 1),
        "embedding_norms": norm_summary,
    }
    with open(args.verification_json, "w") as stream:
        json.dump(verification, stream, indent=2)


if __name__ == "__main__":
    main()
