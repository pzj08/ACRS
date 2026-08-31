import argparse
import csv
import re
from pathlib import Path


SYSTEMS = {
    "acrs_seed3409": "Full ACRS",
    "ablation_no_age_loss_seed3407": "no_age_loss",
    "ablation_no_age_conditioning_seed3407": "no_age_conditioning",
    "ablation_random_gate_init_seed3407": "random_gate_init",
}
PROTOCOLS = {
    "mean_subtracted.txt": "mean_subtracted",
    "no_mean_subtraction.txt": "no_mean_subtraction",
}


def parse(path):
    current = None
    values = {}
    with open(path) as stream:
        for line in stream:
            header = re.match(r"---- (.+)\.kaldi\.score -----", line.strip())
            if header:
                current = header.group(1)
            elif line.startswith("EER"):
                values.setdefault(current, {})["EER"] = float(line.split("=")[1])
            elif line.startswith("minDCF"):
                values.setdefault(current, {})["minDCF"] = float(line.rsplit("=", 1)[1])
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    all_values = {}
    root = Path(args.results_root)
    for directory, system in SYSTEMS.items():
        for filename, protocol in PROTOCOLS.items():
            all_values[(system, protocol)] = parse(root / directory / filename)
    rows = []
    for protocol in PROTOCOLS.values():
        full = all_values[("Full ACRS", protocol)]
        for system in SYSTEMS.values():
            for dataset, values in all_values[(system, protocol)].items():
                reduction = (values["EER"] - full[dataset]["EER"]
                             if system != "Full ACRS" else 0.0)
                relative = (100.0 * reduction / values["EER"]
                            if system != "Full ACRS" else 0.0)
                rows.append({
                    "protocol": protocol, "dataset": dataset,
                    "system": system, "EER": values["EER"],
                    "minDCF": values["minDCF"],
                    "full_absolute_EER_reduction": reduction,
                    "full_relative_EER_reduction_percent": relative,
                })
    with open(args.output, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {args.output}: {len(rows)} rows")


if __name__ == "__main__":
    main()
