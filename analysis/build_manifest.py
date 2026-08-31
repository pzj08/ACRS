import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import yaml


TRIAL_NAMES = (
    "only_ca5.kaldi",
    "only_ca10.kaldi",
    "only_ca15.kaldi",
    "only_ca20.kaldi",
    "vox_ca5.kaldi",
    "vox_ca10.kaldi",
    "vox_ca15.kaldi",
    "vox_ca20.kaldi",
)

def segment_key(utt):
    speaker, video, _ = utt.split("/", 2)
    return f"{speaker}-{video}"


def stable_int(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def age_group(age, boundaries):
    for index, boundary in enumerate(boundaries):
        if age < boundary:
            return index
    return len(boundaries)


def select_shuffle(target, by_group, records):
    groups = sorted(by_group)
    start = stable_int("shuffle-group:" + target["utt_id"]) % (len(groups) - 1)
    candidates = [g for g in groups if g != target["age_group"]]
    for group_offset in range(len(candidates)):
        group = candidates[(start + group_offset) % len(candidates)]
        pool = by_group[group]
        offset = stable_int("shuffle-utt:" + target["utt_id"]) % len(pool)
        for step in range(min(len(pool), 1024)):
            donor = records[pool[(offset + step) % len(pool)]]
            if donor["speaker_id"] != target["speaker_id"]:
                return donor
    raise RuntimeError(f"no shuffle donor for {target['utt_id']}")


def select_far(target, age_order, records):
    left = 0
    right = len(age_order) - 1
    for _ in range(min(len(age_order), 4096)):
        low = records[age_order[left]]
        high = records[age_order[right]]
        donor = max((low, high), key=lambda x: abs(x["age"] - target["age"]))
        if donor["speaker_id"] != target["speaker_id"]:
            return donor
        if donor is low:
            left += 1
        else:
            right -= 1
    raise RuntimeError(f"no far-age donor for {target['utt_id']}")


def select_near(target_index, age_order, age_position, records):
    target = records[target_index]
    position = age_position[target_index]
    candidates = []
    left = position - 1
    while left >= 0:
        donor = records[age_order[left]]
        if donor["speaker_id"] != target["speaker_id"]:
            candidates.append(donor)
            break
        left -= 1
    right = position + 1
    while right < len(age_order):
        donor = records[age_order[right]]
        if donor["speaker_id"] != target["speaker_id"]:
            candidates.append(donor)
            break
        right += 1
    if not candidates:
        raise RuntimeError(f"no near-age donor for {target['utt_id']}")
    return min(candidates, key=lambda donor: (
        abs(donor["age"] - target["age"]), donor["utt_id"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--trials-dir", required=True)
    parser.add_argument("--wav-scp", required=True)
    parser.add_argument("--age-metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    cache = output / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    boundaries = [float(x) for x in
                  config["model_args"]["acsm_args"]["age_bins"]]
    loaded = np.load(args.age_metadata, allow_pickle=True)
    ages = loaded.item() if loaded.shape == () else loaded
    if not isinstance(ages, dict):
        raise TypeError("age metadata must contain a dictionary")

    wav_paths = {}
    with open(args.wav_scp) as stream:
        for line in stream:
            utt, path = line.rstrip().split(maxsplit=1)
            wav_paths[utt] = path

    utterances = set()
    positive_pairs = set()
    trial_counts = {}
    for name in TRIAL_NAMES:
        path = Path(args.trials_dir) / name
        count = 0
        with path.open() as stream:
            for line in stream:
                first, second, label = line.split()
                count += 1
                utterances.add(first)
                utterances.add(second)
                if label == "target":
                    positive_pairs.add(tuple(sorted((first, second))))
        trial_counts[name] = count

    records = []
    missing_age = []
    missing_wav = []
    for utt in sorted(utterances):
        key = segment_key(utt)
        if key not in ages:
            missing_age.append(utt)
            continue
        if utt not in wav_paths:
            missing_wav.append(utt)
            continue
        age = float(ages[key])
        records.append({
            "utt_id": utt,
            "speaker_id": utt.split("/", 1)[0],
            "age": age,
            "age_group": age_group(age, boundaries),
            "wav_path": wav_paths[utt],
        })
    if missing_age or missing_wav:
        raise RuntimeError(
            f"missing age={len(missing_age)} wav={len(missing_wav)}")

    index = {record["utt_id"]: i for i, record in enumerate(records)}
    by_group = {}
    for i, record in enumerate(records):
        by_group.setdefault(record["age_group"], []).append(i)
    age_order = sorted(range(len(records)),
                       key=lambda i: (records[i]["age"], records[i]["utt_id"]))
    age_position = {record_index: position
                    for position, record_index in enumerate(age_order)}

    with (cache / "utterance_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "utt_id", "speaker_id", "age", "age_group", "wav_path"))
        writer.writeheader()
        writer.writerows(records)
    with (cache / "evaluation_keys.list").open("w") as stream:
        for record in records:
            stream.write(record["utt_id"] + "\n")
    with (cache / "positive_pairs.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("utt1", "utt2", "speaker_id", "age1", "age2",
                         "age_gap"))
        for first, second in sorted(positive_pairs):
            left = records[index[first]]
            right = records[index[second]]
            writer.writerow((first, second, left["speaker_id"], left["age"],
                             right["age"], abs(left["age"] - right["age"])))

    donor_rows = []
    for target_index, target in enumerate(records):
        shuffle = select_shuffle(target, by_group, records)
        near = select_near(target_index, age_order, age_position, records)
        far = select_far(target, age_order, records)
        for intervention in ("Shuffle-S3", "Shuffle-S4", "Shuffle-Both"):
            donor_rows.append((target, shuffle, intervention))
        donor_rows.append((target, near, "Near-Age"))
        donor_rows.append((target, far, "Far-Age"))
    with (output / "intervention_donor_mapping.csv").open(
            "w", newline="") as stream:
        fields = (
            "target_utt", "target_speaker", "target_age",
            "target_age_group", "donor_utt", "donor_speaker", "donor_age",
            "donor_age_group", "age_difference", "intervention_type")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for target, donor, intervention in donor_rows:
            writer.writerow({
                "target_utt": target["utt_id"],
                "target_speaker": target["speaker_id"],
                "target_age": target["age"],
                "target_age_group": target["age_group"],
                "donor_utt": donor["utt_id"],
                "donor_speaker": donor["speaker_id"],
                "donor_age": donor["age"],
                "donor_age_group": donor["age_group"],
                "age_difference": abs(target["age"] - donor["age"]),
                "intervention_type": intervention,
            })

    print("unique_utterances", len(records))
    print("unique_positive_pairs", len(positive_pairs))
    print("age_boundaries", boundaries)
    print("trial_counts", trial_counts)


if __name__ == "__main__":
    main()
