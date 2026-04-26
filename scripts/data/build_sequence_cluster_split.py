import argparse
import csv
import json
import random
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import lmdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.reaction_utils import iter_lmdb_json_records, normalize_whitespace, sequence_sha1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build exact / mmseqs splits plus LMDB stores for reaction-to-sequence training."
    )
    parser.add_argument("--mdb_dir", required=True, help="Directory containing the source data.mdb.")
    parser.add_argument(
        "--reaction_smiles_path",
        required=True,
        help="Path to reaction_smiles.tsv or reaction_smiles.jsonl from the resolver script.",
    )
    parser.add_argument("--out_dir", required=True, help="Output directory.")
    parser.add_argument(
        "--mode",
        default="exact_then_mmseqs",
        choices=["exact", "exact_then_mmseqs", "mmseqs"],
        help="Clustering mode.",
    )
    parser.add_argument("--max_records", type=int, default=None, help="Optional development cap.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--valid_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--progress_every", type=int, default=100000)
    parser.add_argument("--commit_every", type=int, default=5000)
    parser.add_argument("--min_sequence_length", type=int, default=10)
    parser.add_argument("--max_sequence_length", type=int, default=4096)
    parser.add_argument(
        "--allow_partial_reactions",
        action="store_true",
        help="Keep reactions whose resolver status is partial as long as reaction_smiles exists.",
    )
    parser.add_argument("--mmseqs_bin", default="mmseqs")
    parser.add_argument("--mmseqs_min_seq_id", type=float, default=0.3)
    parser.add_argument("--mmseqs_coverage", type=float, default=0.8)
    parser.add_argument("--mmseqs_cov_mode", type=int, default=1)
    return parser.parse_args()


def ensure_sqlite_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS entries (
            entry_id TEXT PRIMARY KEY,
            seq_hash TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_entries_seq_hash ON entries (seq_hash);
        CREATE TABLE IF NOT EXISTS sequence_clusters (
            seq_hash TEXT PRIMARY KEY,
            sequence TEXT NOT NULL,
            count INTEGER NOT NULL,
            final_cluster_id TEXT
        );
        """
    )
    conn.commit()


def ensure_reaction_lookup_db(source_path: Path, db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reaction_lookup (
            reaction TEXT PRIMARY KEY,
            reaction_smiles TEXT,
            status TEXT
        )
        """
    )
    row_count = conn.execute("SELECT COUNT(*) FROM reaction_lookup").fetchone()[0]
    if row_count > 0:
        return conn

    batch_rows = []

    def flush_rows():
        if not batch_rows:
            return
        conn.executemany(
            """
            INSERT OR REPLACE INTO reaction_lookup (reaction, reaction_smiles, status)
            VALUES (?, ?, ?)
            """,
            batch_rows,
        )
        conn.commit()
        batch_rows.clear()

    if source_path.suffix == ".jsonl" or source_path.name.endswith(".jsonl.gz"):
        open_fn = open
        if source_path.name.endswith(".gz"):
            import gzip

            open_fn = gzip.open
        with open_fn(source_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                batch_rows.append(
                    (
                        payload["reaction"],
                        payload.get("reaction_smiles"),
                        payload.get("status", "unknown"),
                    )
                )
                if len(batch_rows) >= 10000:
                    flush_rows()
    else:
        dialect = csv.excel_tab if source_path.suffix == ".tsv" else csv.excel
        with source_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, dialect=dialect)
            for row in reader:
                batch_rows.append(
                    (
                        row["reaction"],
                        row.get("reaction_smiles"),
                        row.get("status", "unknown"),
                    )
                )
                if len(batch_rows) >= 10000:
                    flush_rows()
    flush_rows()
    return conn


def open_lmdb_writer(path: Path) -> lmdb.Environment:
    path.mkdir(parents=True, exist_ok=True)
    return lmdb.open(str(path), map_size=1024**4, subdir=True, lock=False)


def flush_entry_rows(
    conn: sqlite3.Connection,
    entry_rows: list[tuple[str, str]],
    sequence_rows: dict,
) -> None:
    if entry_rows:
        conn.executemany(
            "INSERT OR REPLACE INTO entries (entry_id, seq_hash) VALUES (?, ?)",
            entry_rows,
        )
    if sequence_rows:
        conn.executemany(
            """
            INSERT INTO sequence_clusters (seq_hash, sequence, count, final_cluster_id)
            VALUES (?, ?, ?, NULL)
            ON CONFLICT(seq_hash) DO UPDATE SET
                count = sequence_clusters.count + excluded.count
            """,
            [
                (seq_hash, row["sequence"], row["count"])
                for seq_hash, row in sequence_rows.items()
            ],
        )
    conn.commit()
    entry_rows.clear()
    sequence_rows.clear()


def export_unique_sequences_fasta(conn: sqlite3.Connection, fasta_path: Path) -> None:
    with fasta_path.open("w", encoding="utf-8") as handle:
        for seq_hash, sequence in conn.execute(
            "SELECT seq_hash, sequence FROM sequence_clusters ORDER BY seq_hash"
        ):
            handle.write(f">{seq_hash}\n{sequence}\n")


def run_mmseqs_clustering(
    fasta_path: Path,
    out_dir: Path,
    mmseqs_bin: str,
    min_seq_id: float,
    coverage: float,
    cov_mode: int,
) -> dict[str, str]:
    mmseqs_path = shutil.which(mmseqs_bin)
    if mmseqs_path is None:
        raise FileNotFoundError(mmseqs_bin)

    prefix = out_dir / "mmseqs_linclust"
    tmp_dir = out_dir / "mmseqs_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    command = [
        mmseqs_path,
        "easy-linclust",
        str(fasta_path),
        str(prefix),
        str(tmp_dir),
        "--min-seq-id",
        str(min_seq_id),
        "-c",
        str(coverage),
        "--cov-mode",
        str(cov_mode),
    ]
    subprocess.run(command, check=True)

    cluster_tsv = prefix.with_name(prefix.name + "_cluster.tsv")
    if not cluster_tsv.exists():
        candidates = list(out_dir.glob("mmseqs_linclust*_cluster.tsv"))
        if not candidates:
            raise FileNotFoundError("mmseqs cluster TSV output not found")
        cluster_tsv = candidates[0]

    cluster_map = {}
    with cluster_tsv.open("r", encoding="utf-8") as handle:
        for line in handle:
            rep_id, member_id = line.rstrip("\n").split("\t")
            cluster_map[member_id] = rep_id
    return cluster_map


def assign_cluster_splits(cluster_ids: list[str], seed: int, valid_ratio: float, test_ratio: float) -> dict[str, str]:
    rng = random.Random(seed)
    shuffled = cluster_ids[:]
    rng.shuffle(shuffled)

    total = len(shuffled)
    test_count = int(total * test_ratio)
    valid_count = int(total * valid_ratio)
    if total > 0 and test_ratio > 0 and test_count == 0:
        test_count = 1
    if total - test_count > 0 and valid_ratio > 0 and valid_count == 0:
        valid_count = 1
    if test_count + valid_count > total:
        valid_count = max(0, total - test_count)

    assignments = {}
    for cluster_id in shuffled[:test_count]:
        assignments[cluster_id] = "test"
    for cluster_id in shuffled[test_count : test_count + valid_count]:
        assignments[cluster_id] = "valid"
    for cluster_id in shuffled[test_count + valid_count :]:
        assignments[cluster_id] = "train"
    return assignments


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_dir = out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = out_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    lookup_conn = ensure_reaction_lookup_db(
        Path(args.reaction_smiles_path), out_dir / "reaction_lookup.sqlite"
    )
    dataset_conn = sqlite3.connect(out_dir / "dataset_index.sqlite")
    ensure_sqlite_tables(dataset_conn)

    seq_env = open_lmdb_writer(processed_dir / "LMDB_seqonly")
    reaction_env = open_lmdb_writer(processed_dir / "LMDB_reaction_smiles")
    raw_reaction_env = open_lmdb_writer(processed_dir / "LMDB_raw_reaction")
    source_env = open_lmdb_writer(processed_dir / "LMDB_source")

    seq_txn = seq_env.begin(write=True)
    reaction_txn = reaction_env.begin(write=True)
    raw_reaction_txn = raw_reaction_env.begin(write=True)
    source_txn = source_env.begin(write=True)

    entry_rows = []
    sequence_rows = {}
    processed_records = 0
    skipped_records = 0
    unresolved_reactions = 0
    started_at = time.time()

    for entry_id, payload in iter_lmdb_json_records(args.mdb_dir):
        if args.max_records is not None and processed_records >= args.max_records:
            break

        sequence = str(payload.get("sequence", "")).strip()
        raw_reaction = normalize_whitespace(str(payload.get("reaction", "")))
        source = normalize_whitespace(str(payload.get("source", "unknown")))
        if not sequence or not raw_reaction:
            skipped_records += 1
            continue
        if not (args.min_sequence_length <= len(sequence) <= args.max_sequence_length):
            skipped_records += 1
            continue

        lookup_row = lookup_conn.execute(
            "SELECT reaction_smiles, status FROM reaction_lookup WHERE reaction = ?",
            (raw_reaction,),
        ).fetchone()
        if lookup_row is None:
            skipped_records += 1
            unresolved_reactions += 1
            continue

        reaction_smiles, status = lookup_row
        if reaction_smiles is None:
            skipped_records += 1
            unresolved_reactions += 1
            continue
        if status != "resolved" and not args.allow_partial_reactions:
            skipped_records += 1
            unresolved_reactions += 1
            continue

        seq_hash = sequence_sha1(sequence)
        entry_rows.append((entry_id, seq_hash))
        if seq_hash not in sequence_rows:
            sequence_rows[seq_hash] = {"sequence": sequence, "count": 1}
        else:
            sequence_rows[seq_hash]["count"] += 1

        entry_key = entry_id.encode("utf-8")
        seq_txn.put(entry_key, sequence.encode("utf-8"))
        reaction_txn.put(entry_key, reaction_smiles.encode("utf-8"))
        raw_reaction_txn.put(entry_key, raw_reaction.encode("utf-8"))
        source_txn.put(entry_key, source.encode("utf-8"))

        processed_records += 1
        if processed_records % args.commit_every == 0:
            flush_entry_rows(dataset_conn, entry_rows, sequence_rows)
            seq_txn.commit()
            reaction_txn.commit()
            raw_reaction_txn.commit()
            source_txn.commit()
            seq_txn = seq_env.begin(write=True)
            reaction_txn = reaction_env.begin(write=True)
            raw_reaction_txn = raw_reaction_env.begin(write=True)
            source_txn = source_env.begin(write=True)

        if processed_records % args.progress_every == 0:
            elapsed = time.time() - started_at
            rps = processed_records / elapsed if elapsed > 0 else 0.0
            print(
                f"[build] kept_records={processed_records} skipped={skipped_records} "
                f"unresolved={unresolved_reactions} rps={rps:.2f}"
            )

    flush_entry_rows(dataset_conn, entry_rows, sequence_rows)
    seq_txn.commit()
    reaction_txn.commit()
    raw_reaction_txn.commit()
    source_txn.commit()
    seq_env.close()
    reaction_env.close()
    raw_reaction_env.close()
    source_env.close()

    unique_fasta_path = processed_dir / "unique_sequences.fasta"
    export_unique_sequences_fasta(dataset_conn, unique_fasta_path)

    effective_mode = args.mode
    cluster_map = {}
    if args.mode in {"exact_then_mmseqs", "mmseqs"}:
        try:
            cluster_map = run_mmseqs_clustering(
                fasta_path=unique_fasta_path,
                out_dir=processed_dir,
                mmseqs_bin=args.mmseqs_bin,
                min_seq_id=args.mmseqs_min_seq_id,
                coverage=args.mmseqs_coverage,
                cov_mode=args.mmseqs_cov_mode,
            )
        except FileNotFoundError:
            if args.mode == "mmseqs":
                raise
            effective_mode = "exact"
        except subprocess.CalledProcessError:
            if args.mode == "mmseqs":
                raise
            effective_mode = "exact"

    if effective_mode == "exact":
        dataset_conn.execute(
            "UPDATE sequence_clusters SET final_cluster_id = seq_hash WHERE final_cluster_id IS NULL"
        )
    else:
        update_rows = []
        for seq_hash, in dataset_conn.execute("SELECT seq_hash FROM sequence_clusters"):
            update_rows.append((cluster_map.get(seq_hash, seq_hash), seq_hash))
        dataset_conn.executemany(
            "UPDATE sequence_clusters SET final_cluster_id = ? WHERE seq_hash = ?",
            update_rows,
        )
    dataset_conn.commit()

    cluster_ids = [
        row[0]
        for row in dataset_conn.execute(
            "SELECT DISTINCT final_cluster_id FROM sequence_clusters ORDER BY final_cluster_id"
        )
    ]
    cluster_to_split = assign_cluster_splits(
        cluster_ids=cluster_ids,
        seed=args.seed,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
    )

    train_split_path = split_dir / "train_repid-entryid.tsv"
    valid_split_path = split_dir / "valid_repid-entryid.tsv"
    test_split_path = split_dir / "test_repid-entryid.tsv"
    train_cluster_env = open_lmdb_writer(processed_dir / "LMDB_train_cluster")
    train_cluster_txn = train_cluster_env.begin(write=True)

    split_handles = {
        "train": train_split_path.open("w", encoding="utf-8"),
        "valid": valid_split_path.open("w", encoding="utf-8"),
        "test": test_split_path.open("w", encoding="utf-8"),
    }

    split_counts = defaultdict(int)
    grouped_query = dataset_conn.execute(
        """
        SELECT sc.final_cluster_id, e.entry_id
        FROM entries e
        JOIN sequence_clusters sc ON e.seq_hash = sc.seq_hash
        ORDER BY sc.final_cluster_id, e.entry_id
        """
    )

    current_cluster_id = None
    current_entry_ids = []
    flushed_clusters = 0

    def flush_cluster(cluster_id: str, entry_ids: list[str]) -> None:
        nonlocal flushed_clusters, train_cluster_txn
        if cluster_id is None or not entry_ids:
            return
        split_name = cluster_to_split[cluster_id]
        for entry_id in entry_ids:
            split_handles[split_name].write(f"{cluster_id}\t{entry_id}\n")
            split_counts[split_name] += 1
        if split_name == "train":
            train_cluster_txn.put(
                cluster_id.encode("utf-8"),
                json.dumps(entry_ids, ensure_ascii=False).encode("utf-8"),
            )
            flushed_clusters += 1
            if flushed_clusters % 1000 == 0:
                train_cluster_txn.commit()
                train_cluster_txn = train_cluster_env.begin(write=True)

    for cluster_id, entry_id in grouped_query:
        if current_cluster_id is None:
            current_cluster_id = cluster_id
        if cluster_id != current_cluster_id:
            flush_cluster(current_cluster_id, current_entry_ids)
            current_cluster_id = cluster_id
            current_entry_ids = []
        current_entry_ids.append(entry_id)
    flush_cluster(current_cluster_id, current_entry_ids)

    train_cluster_txn.commit()
    train_cluster_env.close()
    for handle in split_handles.values():
        handle.close()

    summary = {
        "mdb_dir": args.mdb_dir,
        "reaction_smiles_path": args.reaction_smiles_path,
        "effective_mode": effective_mode,
        "processed_records": processed_records,
        "skipped_records": skipped_records,
        "unresolved_reactions": unresolved_reactions,
        "unique_exact_sequences": dataset_conn.execute(
            "SELECT COUNT(*) FROM sequence_clusters"
        ).fetchone()[0],
        "train_entries": split_counts["train"],
        "valid_entries": split_counts["valid"],
        "test_entries": split_counts["test"],
        "train_split_path": str(train_split_path),
        "valid_split_path": str(valid_split_path),
        "test_split_path": str(test_split_path),
        "processed_dir": str(processed_dir),
        "elapsed_seconds": round(time.time() - started_at, 2),
    }
    with (out_dir / "build_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    dataset_conn.close()
    lookup_conn.close()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
