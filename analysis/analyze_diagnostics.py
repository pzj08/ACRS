import argparse
import csv
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr


AGE_LABELS = ("<21", "21–30", "31–40", "41–50", "51–60", "61–70",
              "≥71")
BIN_LABELS = ("[0,5)", "[5,10)", "[10,15)", "[15,20)", "[20,+inf)")
BIN_EDGES = np.array([0.0, 5.0, 10.0, 15.0, 20.0, np.inf])


def poisson_ci(values, rng, repetitions=1000, chunk=25):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan"), float("nan")
    estimates = []
    for start in range(0, repetitions, chunk):
        count = min(chunk, repetitions - start)
        weights = rng.poisson(1.0, size=(count, len(values))).astype(np.float64)
        denominator = weights.sum(axis=1)
        estimates.extend((weights @ values / np.maximum(denominator, 1)).tolist())
    return tuple(np.percentile(estimates, [2.5, 97.5]))


def poisson_correlation_ci(x, y, rng, repetitions=1000, chunk=20):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    from scipy.stats import rankdata
    rx = rankdata(x).astype(np.float32)
    ry = rankdata(y).astype(np.float32)
    rx = (rx - rx.mean()) / rx.std()
    ry = (ry - ry.mean()) / ry.std()
    rx2 = rx * rx
    ry2 = ry * ry
    rxy = rx * ry
    estimates = []
    for start in range(0, repetitions, chunk):
        count = min(chunk, repetitions - start)
        weights = rng.poisson(1.0, size=(count, len(x))).astype(np.float32)
        total = np.maximum(weights.sum(axis=1), 1.0)
        sum_x = weights @ rx
        sum_y = weights @ ry
        covariance = weights @ rxy - sum_x * sum_y / total
        variance_x = weights @ rx2 - sum_x * sum_x / total
        variance_y = weights @ ry2 - sum_y * sum_y / total
        estimates.extend((covariance / np.sqrt(variance_x * variance_y)).tolist())
    return tuple(np.percentile(estimates, [2.5, 97.5]))


def group_summaries(values, groups, rng):
    rows = []
    for group, label in enumerate(AGE_LABELS):
        subset = values[groups == group]
        low, high = poisson_ci(subset, rng)
        rows.append({
            "age_group": group, "age_range": label, "N": len(subset),
            "mean": float(subset.mean()), "std": float(subset.std()),
            "median": float(np.median(subset)), "ci95_low": low,
            "ci95_high": high,
        })
    return rows


def save_gate_specialization(gates, groups, stage, output, figure_dir):
    centroids = np.stack([gates[groups == group].mean(axis=0)
                          for group in range(len(AGE_LABELS))])
    between_variance = centroids.var(axis=0)
    top = np.argsort(between_variance)[::-1][:16]
    distances = [np.abs(centroids[first] - centroids[second]).mean()
                 for first, second in combinations(range(len(AGE_LABELS)), 2)]
    fields = ["age_group", "age_range", "N", "gate_mean", "gate_std",
              "top16_channels", "maximum_centroid_pairwise_L1",
              "mean_centroid_pairwise_L1"] + [
                  f"channel_{index:03d}_centroid"
                  for index in range(gates.shape[1])] + [
                  f"channel_{index:03d}_between_age_variance"
                  for index in range(gates.shape[1])]
    with open(output, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for group, label in enumerate(AGE_LABELS):
            subset = gates[groups == group]
            row = {
                "age_group": group, "age_range": label, "N": len(subset),
                "gate_mean": float(subset.mean()), "gate_std": float(subset.std()),
                "top16_channels": ";".join(map(str, top.tolist())),
                "maximum_centroid_pairwise_L1": max(distances),
                "mean_centroid_pairwise_L1": float(np.mean(distances)),
            }
            row.update({f"channel_{index:03d}_centroid": centroids[group, index]
                        for index in range(gates.shape[1])})
            row.update({f"channel_{index:03d}_between_age_variance":
                        between_variance[index]
                        for index in range(gates.shape[1])})
            writer.writerow(row)

    matrix = centroids[:, top]
    fig, axis = plt.subplots(figsize=(5.4, 2.55))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis")
    axis.set_xlabel("Gate channel (ranked by between-age variance)")
    axis.set_ylabel("Age group (years)")
    axis.set_xticks(range(len(top)), [str(value) for value in top],
                    rotation=60, ha="right", fontsize=7)
    axis.set_yticks(range(len(AGE_LABELS)), AGE_LABELS)
    colorbar = fig.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Mean gate activation")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(figure_dir / f"gate_age_heatmap_stage{stage}.{suffix}",
                    dpi=400, bbox_inches="tight")
    plt.close(fig)
    return {
        "top_channels": top.tolist(), "between_variance": between_variance[top].tolist(),
        "max_centroid_l1": max(distances),
        "mean_centroid_l1": float(np.mean(distances)),
        "group_overall_mean": [float(gates[groups == group].mean())
                               for group in range(len(AGE_LABELS))],
    }


def pair_gate_distances(gates, left, right, chunk=20000):
    l1 = np.empty(len(left), dtype=np.float32)
    cosine = np.empty(len(left), dtype=np.float32)
    norms = np.linalg.norm(gates, axis=1)
    for start in range(0, len(left), chunk):
        end = min(start + chunk, len(left))
        first = gates[left[start:end]]
        second = gates[right[start:end]]
        l1[start:end] = np.abs(first - second).mean(axis=1)
        cosine[start:end] = 1.0 - np.einsum("ij,ij->i", first, second) / (
            norms[left[start:end]] * norms[right[start:end]])
    return l1, cosine


def bin_statistics(values, gaps, stage, measure, rng):
    rows = []
    assignments = np.digitize(gaps, BIN_EDGES[1:-1], right=False)
    rho, p_value = spearmanr(gaps, values)
    rho_low, rho_high = poisson_correlation_ci(gaps, values, rng)
    for bin_index, label in enumerate(BIN_LABELS):
        subset = values[assignments == bin_index]
        low, high = poisson_ci(subset, rng)
        rows.append({
            "stage": stage, "measure": measure, "age_gap_bin": label,
            "N": len(subset), "mean": float(subset.mean()),
            "std": float(subset.std()), "median": float(np.median(subset)),
            "ci95_low": low, "ci95_high": high,
            "spearman_rho_all_pairs": float(rho),
            "spearman_p_value_all_pairs": float(p_value),
            "spearman_ci95_low": rho_low, "spearman_ci95_high": rho_high,
            "bootstrap": "Poisson bootstrap, 1000 replicates",
        })
    return rows


def write_rows(path, rows):
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--positive-pairs", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260831)

    cache = torch.load(args.cache, map_location="cpu")
    utterances = cache["utt_ids"]
    index = {utt: position for position, utt in enumerate(utterances)}
    groups = cache["age_groups"].numpy()
    ages = cache["ages"].numpy()
    gate3 = cache["gate3"].numpy()
    gate4 = cache["gate4"].numpy()
    specialization3 = save_gate_specialization(
        gate3, groups, 3, output_dir / "gate_age_specialization_stage3.csv",
        figure_dir)
    specialization4 = save_gate_specialization(
        gate4, groups, 4, output_dir / "gate_age_specialization_stage4.csv",
        figure_dir)

    predictions = cache["age_predictions_years"].numpy()
    prediction_rho, prediction_p = spearmanr(ages, predictions)
    prediction_rows = [{
        "scope": "all", "age_group": "", "age_range": "all",
        "N": len(ages), "mean_true_age": float(ages.mean()),
        "mean_predicted_age": float(predictions.mean()),
        "MAE_years": float(np.abs(predictions - ages).mean()),
        "spearman_rho": float(prediction_rho),
        "spearman_p_value": float(prediction_p),
    }]
    for group, label in enumerate(AGE_LABELS):
        selector = groups == group
        prediction_rows.append({
            "scope": "age_group", "age_group": group, "age_range": label,
            "N": int(selector.sum()), "mean_true_age": float(ages[selector].mean()),
            "mean_predicted_age": float(predictions[selector].mean()),
            "MAE_years": float(np.abs(predictions[selector] - ages[selector]).mean()),
            "spearman_rho": "", "spearman_p_value": "",
        })
    write_rows(output_dir / "age_prediction_statistics.csv", prediction_rows)

    pair_rows = []
    with open(args.positive_pairs) as stream:
        pair_rows = list(csv.DictReader(stream))
    left = np.fromiter((index[row["utt1"]] for row in pair_rows), dtype=np.int64)
    right = np.fromiter((index[row["utt2"]] for row in pair_rows), dtype=np.int64)
    gaps = np.abs(ages[left] - ages[right])
    d3_l1, d3_cos = pair_gate_distances(gate3, left, right)
    d4_l1, d4_cos = pair_gate_distances(gate4, left, right)

    pair_output = output_dir / "gate_distance_vs_age_gap.csv"
    with open(pair_output, "w", newline="") as stream:
        fields = ["speaker_id", "utt1", "utt2", "age1", "age2", "age_gap",
                  "age_gap_bin", "D3_L1", "D4_L1", "D3_cosine_distance",
                  "D4_cosine_distance"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        assignments = np.digitize(gaps, BIN_EDGES[1:-1], right=False)
        for number, row in enumerate(pair_rows):
            writer.writerow({
                "speaker_id": row["speaker_id"], "utt1": row["utt1"],
                "utt2": row["utt2"], "age1": ages[left[number]],
                "age2": ages[right[number]], "age_gap": gaps[number],
                "age_gap_bin": BIN_LABELS[assignments[number]],
                "D3_L1": d3_l1[number], "D4_L1": d4_l1[number],
                "D3_cosine_distance": d3_cos[number],
                "D4_cosine_distance": d4_cos[number],
            })
    statistics = []
    for stage, measure, values in (
            (3, "mean_absolute_channel_difference", d3_l1),
            (4, "mean_absolute_channel_difference", d4_l1),
            (3, "cosine_distance", d3_cos), (4, "cosine_distance", d4_cos)):
        statistics.extend(bin_statistics(values, gaps, stage, measure, rng))
    write_rows(output_dir / "gate_distance_statistics.csv", statistics)

    l1_stats = [row for row in statistics
                if row["measure"] == "mean_absolute_channel_difference"]
    fig, axis = plt.subplots(figsize=(3.45, 2.55))
    x = np.arange(len(BIN_LABELS))
    for stage, marker, linestyle in ((3, "o", "-"), (4, "s", "--")):
        rows = [row for row in l1_stats if row["stage"] == stage]
        means = np.array([row["mean"] for row in rows])
        lower = means - np.array([row["ci95_low"] for row in rows])
        upper = np.array([row["ci95_high"] for row in rows]) - means
        axis.errorbar(x, means, yerr=np.stack((lower, upper)), marker=marker,
                      linestyle=linestyle, color="black", capsize=2,
                      linewidth=1.2, markersize=4, label=f"Stage {stage}")
    axis.set_xticks(x, ("0–5", "5–10", "10–15", "15–20", "≥20"))
    axis.set_xlabel("Within-speaker age gap (years)")
    axis.set_ylabel("Mean per-channel gate distance")
    axis.legend(frameon=False)
    axis.grid(axis="y", linewidth=0.35, alpha=0.4)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(figure_dir / f"gate_distance_vs_age_gap.{suffix}",
                    dpi=400, bbox_inches="tight")
    plt.close(fig)

    suppression3 = cache["suppression_ratio_stage3"].numpy()
    suppression4 = cache["suppression_ratio_stage4"].numpy()
    suppression_group_rows = []
    for stage, values in ((3, suppression3), (4, suppression4)):
        for row in group_summaries(values, groups, rng):
            row = {"stage": stage, **row}
            suppression_group_rows.append(row)
    write_rows(output_dir / "suppression_ratio_by_age.csv",
               suppression_group_rows)

    suppression_pair_rows = []
    pair_summary = {}
    for stage, values in ((3, suppression3), (4, suppression4)):
        means = (values[left] + values[right]) / 2.0
        differences = np.abs(values[left] - values[right])
        mean_rho, mean_p = spearmanr(gaps, means)
        difference_rho, difference_p = spearmanr(gaps, differences)
        mean_ci = poisson_correlation_ci(gaps, means, rng)
        difference_ci = poisson_correlation_ci(gaps, differences, rng)
        pair_summary[stage] = (means, differences)
        assignments = np.digitize(gaps, BIN_EDGES[1:-1], right=False)
        for bin_index, label in enumerate(BIN_LABELS):
            selector = assignments == bin_index
            suppression_pair_rows.append({
                "stage": stage, "age_gap_bin": label,
                "N": int(selector.sum()),
                "pair_suppression_mean": float(means[selector].mean()),
                "pair_suppression_mean_std": float(means[selector].std()),
                "pair_suppression_difference_mean":
                    float(differences[selector].mean()),
                "pair_suppression_difference_std":
                    float(differences[selector].std()),
                "rho_age_gap_pair_mean": float(mean_rho),
                "p_age_gap_pair_mean": float(mean_p),
                "rho_age_gap_pair_mean_ci95_low": mean_ci[0],
                "rho_age_gap_pair_mean_ci95_high": mean_ci[1],
                "rho_age_gap_pair_difference": float(difference_rho),
                "p_age_gap_pair_difference": float(difference_p),
                "rho_age_gap_pair_difference_ci95_low": difference_ci[0],
                "rho_age_gap_pair_difference_ci95_high": difference_ci[1],
                "bootstrap": "Poisson bootstrap, 1000 replicates",
            })
    write_rows(output_dir / "suppression_ratio_vs_age_gap.csv",
               suppression_pair_rows)

    fig, axis = plt.subplots(figsize=(4.7, 2.65))
    x = np.arange(len(AGE_LABELS))
    for stage, marker, linestyle in ((3, "o", "-"), (4, "s", "--")):
        rows = [row for row in suppression_group_rows if row["stage"] == stage]
        means = np.array([row["mean"] for row in rows])
        lower = means - np.array([row["ci95_low"] for row in rows])
        upper = np.array([row["ci95_high"] for row in rows]) - means
        axis.errorbar(x, means, yerr=np.stack((lower, upper)), marker=marker,
                      linestyle=linestyle, color="black", capsize=2,
                      linewidth=1.2, markersize=4, label=f"Stage {stage}")
    axis.set_xticks(x, AGE_LABELS)
    axis.set_xlabel("Age group (years)")
    axis.set_ylabel("Suppression ratio")
    axis.legend(frameon=False)
    axis.grid(axis="y", linewidth=0.35, alpha=0.4)
    fig.tight_layout()
    fig.savefig(figure_dir / "suppression_ratio_by_age.pdf",
                bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(3.45, 2.55))
    assignments = np.digitize(gaps, BIN_EDGES[1:-1], right=False)
    for stage, marker, linestyle in ((3, "o", "-"), (4, "s", "--")):
        means, _ = pair_summary[stage]
        grouped = np.array([means[assignments == value].mean()
                            for value in range(len(BIN_LABELS))])
        axis.plot(np.arange(len(BIN_LABELS)), grouped, marker=marker,
                  linestyle=linestyle, color="black", linewidth=1.2,
                  markersize=4, label=f"Stage {stage}")
    axis.set_xticks(np.arange(len(BIN_LABELS)),
                    ("0–5", "5–10", "10–15", "15–20", "≥20"))
    axis.set_xlabel("Within-speaker age gap (years)")
    axis.set_ylabel("Mean pair suppression ratio")
    axis.legend(frameon=False)
    axis.grid(axis="y", linewidth=0.35, alpha=0.4)
    fig.tight_layout()
    fig.savefig(figure_dir / "suppression_ratio_vs_age_gap.pdf",
                bbox_inches="tight")
    plt.close(fig)

    torch.save({
        "gate_specialization_stage3": specialization3,
        "gate_specialization_stage4": specialization4,
        "pair_count": len(pair_rows),
    }, output_dir / "cache" / "analysis_summary.pt")


if __name__ == "__main__":
    main()
