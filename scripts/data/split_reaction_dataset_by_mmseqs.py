import argparse
import json
import random
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split reaction dataset by mmseqs2 sequence clusters."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="/jinkedu/enzyme-design/ReactZyme-main/data/positive_saprot.pt",
        help="Input reaction dataset .pt (entry_id -> (substrate, product, seq)).",
    )
    parser.add_argument(
        "--train_satoken_output_path",
        type=str,
        default="/jinkedu/enzyme-design/ReactZyme-main/data/positive_saprot_mmseqs30_train.pt",
        help="Output .pt path for satoken train split.",
    )
    parser.add_argument(
        "--val_satoken_output_path",
        type=str,
        default="/jinkedu/enzyme-design/ReactZyme-main/data/positive_saprot_mmseqs30_val.pt",
        help="Output .pt path for satoken val split.",
    )
    parser.add_argument(
        "--train_sequence_output_path",
        type=str,
        default="/jinkedu/enzyme-design/ReactZyme-main/data/positive_sequence_mmseqs30_train.pt",
        help="Output .pt path for pure-sequence train split.",
    )
    parser.add_argument(
        "--val_sequence_output_path",
        type=str,
        default="/jinkedu/enzyme-design/ReactZyme-main/data/positive_sequence_mmseqs30_val.pt",
        help="Output .pt path for pure-sequence val split.",
    )
    parser.add_argument(
        "--stats_output_path",
        type=str,
        default="/jinkedu/enzyme-design/ReactZyme-main/data/positive_saprot_mmseqs30_split_stats.json",
        help="Output JSON path for split statistics.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Target validation ratio by sample count.",
    )
    parser.add_argument(
        "--min_seq_id",
        type=float,
        default=0.3,
        help="mmseqs2 min sequence identity threshold.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for cluster assignment.",
    )
    parser.add_argument(
        "--tmp_dir",
        type=str,
        default="tmp/mmseqs_split",
        help="Temporary directory for fasta/mmseqs intermediate files.",
    )
    parser.add_argument(
        "--keep_tmp",
        action="store_true",
        help="Keep temporary directory for debugging.",
    )
    parser.add_argument(
        "--cluster_seq_source",
        type=str,
        default="aa_from_satoken",
        choices=["aa_from_satoken", "raw_seq"],
        help=(
            "Sequence source for mmseqs clustering: "
            "'aa_from_satoken' extracts uppercase amino-acid sequence from satoken; "
            "'raw_seq' uses the raw third field directly."
        ),
    )
    return parser.parse_args()


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def extract_aa_sequence_from_satoken(seq: str) -> str:
    aa_seq = "".join(ch for ch in seq if ch.isupper())
    if len(aa_seq) == 0:
        raise ValueError("No uppercase amino-acid characters found in satoken.")
    return aa_seq


def convert_sample_to_sequence(sample_value):
    substrate, product, seq = sample_value[0], sample_value[1], sample_value[-1]
    pure_seq = extract_aa_sequence_from_satoken(seq)
    converted = list(sample_value)
    converted[0] = substrate
    converted[1] = product
    converted[-1] = pure_seq
    return tuple(converted)


def write_fasta(seq_list, fasta_path: Path):
    with fasta_path.open("w") as f:
        for idx, seq in enumerate(seq_list):
            f.write(f">seq_{idx}\n{seq}\n")


def read_mmseqs_clusters(cluster_tsv: Path):
    rep_to_members = defaultdict(list)
    with cluster_tsv.open("r") as f:
        for line in f:
            rep, member = line.rstrip("\n").split("\t")
            rep_to_members[rep].append(member)
    return rep_to_members


def assign_clusters_balanced(clusters, target_val_samples: int, seed: int):
    rng = random.Random(seed)
    shuffled_clusters = list(clusters)
    rng.shuffle(shuffled_clusters)
    shuffled_clusters.sort(key=len, reverse=True)

    val_clusters = []
    train_clusters = []
    val_count = 0

    for cluster_entry_ids in shuffled_clusters:
        cluster_size = len(cluster_entry_ids)
        add_to_val_gap = abs((val_count + cluster_size) - target_val_samples)
        keep_in_train_gap = abs(val_count - target_val_samples)
        if add_to_val_gap <= keep_in_train_gap:
            val_clusters.append(cluster_entry_ids)
            val_count += cluster_size
        else:
            train_clusters.append(cluster_entry_ids)

    if len(train_clusters) == 0 and len(val_clusters) > 1:
        moved_cluster = val_clusters.pop(0)
        train_clusters.append(moved_cluster)
    if len(val_clusters) == 0 and len(train_clusters) > 1:
        moved_cluster = train_clusters.pop()
        val_clusters.append(moved_cluster)

    train_ids = set()
    val_ids = set()
    for cluster_entry_ids in train_clusters:
        train_ids.update(cluster_entry_ids)
    for cluster_entry_ids in val_clusters:
        val_ids.update(cluster_entry_ids)

    return train_ids, val_ids


def main():
    args = parse_args()
    if shutil.which("mmseqs") is None:
        raise RuntimeError(
            "mmseqs2 not found in PATH. Please activate an environment containing mmseqs2."
        )

    input_path = Path(args.input_path)
    train_satoken_output_path = Path(args.train_satoken_output_path)
    val_satoken_output_path = Path(args.val_satoken_output_path)
    train_sequence_output_path = Path(args.train_sequence_output_path)
    val_sequence_output_path = Path(args.val_sequence_output_path)
    stats_output_path = Path(args.stats_output_path)
    tmp_dir = Path(args.tmp_dir)

    ensure_parent(train_satoken_output_path)
    ensure_parent(val_satoken_output_path)
    ensure_parent(train_sequence_output_path)
    ensure_parent(val_sequence_output_path)
    ensure_parent(stats_output_path)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {input_path}")
    data = torch.load(input_path, map_location="cpu")
    if len(data) == 0:
        raise ValueError(f"Input dataset is empty: {input_path}")

    seq_to_entries = defaultdict(list)
    entry_to_sample = {}
    for entry_id, value in data.items():
        if not isinstance(value, (list, tuple)) or len(value) < 3:
            raise ValueError(f"Unsupported sample format for entry {entry_id}: {value}")
        seq = value[-1]
        seq_to_entries[seq].append(str(entry_id))
        entry_to_sample[str(entry_id)] = value

    invalid_satoken_examples = []
    if args.cluster_seq_source == "aa_from_satoken":
        cluster_seq_to_entries = defaultdict(list)
        for raw_seq, entry_ids in seq_to_entries.items():
            try:
                aa_seq = extract_aa_sequence_from_satoken(raw_seq)
            except ValueError:
                if len(invalid_satoken_examples) < 20:
                    invalid_satoken_examples.append(
                        {
                            "entry_id": entry_ids[0],
                            "seq_prefix": raw_seq[:80],
                            "seq_len": len(raw_seq),
                        }
                    )
                continue
            cluster_seq_to_entries[aa_seq].extend(entry_ids)
        if invalid_satoken_examples:
            raise ValueError(
                "Failed to extract amino-acid sequence from some satokens. "
                f"Examples: {invalid_satoken_examples}"
            )
        seq_list = list(cluster_seq_to_entries.keys())
        seq_lookup = cluster_seq_to_entries
    else:
        seq_list = list(seq_to_entries.keys())
        seq_lookup = seq_to_entries

    fasta_path = tmp_dir / "unique_sequences.fasta"
    write_fasta(seq_list, fasta_path)

    cluster_prefix = tmp_dir / "mmseqs_cluster"
    mmseqs_tmp = tmp_dir / "mmseqs_tmp"
    run_cmd(
        [
            "mmseqs",
            "easy-cluster",
            str(fasta_path),
            str(cluster_prefix),
            str(mmseqs_tmp),
            "--min-seq-id",
            str(args.min_seq_id),
        ]
    )

    cluster_tsv = Path(str(cluster_prefix) + "_cluster.tsv")
    if not cluster_tsv.exists():
        raise FileNotFoundError(f"Expected mmseqs cluster file not found: {cluster_tsv}")
    rep_to_members = read_mmseqs_clusters(cluster_tsv)

    clusters = []
    for _, members in rep_to_members.items():
        cluster_entry_ids = []
        for member in members:
            seq_idx = int(member.split("_")[-1])
            seq = seq_list[seq_idx]
            cluster_entry_ids.extend(seq_lookup[seq])
        clusters.append(cluster_entry_ids)

    total_samples = len(entry_to_sample)
    target_val_samples = int(total_samples * args.val_ratio)
    train_ids, val_ids = assign_clusters_balanced(
        clusters=clusters,
        target_val_samples=target_val_samples,
        seed=args.seed,
    )

    train_satoken_data = {eid: entry_to_sample[eid] for eid in train_ids}
    val_satoken_data = {eid: entry_to_sample[eid] for eid in val_ids}
    train_sequence_data = {
        eid: convert_sample_to_sequence(entry_to_sample[eid]) for eid in train_ids
    }
    val_sequence_data = {
        eid: convert_sample_to_sequence(entry_to_sample[eid]) for eid in val_ids
    }

    torch.save(train_satoken_data, train_satoken_output_path)
    torch.save(val_satoken_data, val_satoken_output_path)
    torch.save(train_sequence_data, train_sequence_output_path)
    torch.save(val_sequence_data, val_sequence_output_path)

    stats = {
        "input_path": str(input_path),
        "train_satoken_output_path": str(train_satoken_output_path),
        "val_satoken_output_path": str(val_satoken_output_path),
        "train_sequence_output_path": str(train_sequence_output_path),
        "val_sequence_output_path": str(val_sequence_output_path),
        "min_seq_id": args.min_seq_id,
        "cluster_seq_source": args.cluster_seq_source,
        "val_ratio_target": args.val_ratio,
        "seed": args.seed,
        "total_samples": total_samples,
        "unique_satokens": len(seq_to_entries),
        "unique_cluster_sequences": len(seq_list),
        "cluster_count": len(clusters),
        "train_samples": len(train_satoken_data),
        "val_samples": len(val_satoken_data),
        "val_ratio_actual": len(val_satoken_data) / total_samples,
    }
    with stats_output_path.open("w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))

    if not args.keep_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
