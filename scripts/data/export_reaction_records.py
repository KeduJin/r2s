import argparse
import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.reaction_utils import (
    normalize_compound_name,
    normalize_whitespace,
    parse_stoichiometry,
    sequence_sha1,
    split_reaction_equation,
    write_jsonl,
    iter_lmdb_json_records,
)


class JsonlShardWriter:
    def __init__(self, output_dir: Path, shard_size: int) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.rows_in_current_shard = 0
        self.shard_index = 0
        self.current_rows = []

    def write(self, row: dict) -> None:
        self.current_rows.append(row)
        self.rows_in_current_shard += 1
        if self.rows_in_current_shard >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.current_rows:
            return
        shard_path = self.output_dir / f"records_{self.shard_index:06d}.jsonl.gz"
        write_jsonl(str(shard_path), self.current_rows)
        self.current_rows = []
        self.rows_in_current_shard = 0
        self.shard_index += 1

    def close(self) -> None:
        self.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream scan reaction-sequence LMDB and cache unique reactions / sequences."
    )
    parser.add_argument("--mdb_dir", required=True, help="Directory containing data.mdb and lock.mdb.")
    parser.add_argument("--out_dir", required=True, help="Directory for scan outputs.")
    parser.add_argument("--max_records", type=int, default=None, help="Optional cap for development scans.")
    parser.add_argument("--sample_n", type=int, default=0, help="Number of early records to export as JSONL shards.")
    parser.add_argument("--records_shard_size", type=int, default=100000, help="Rows per JSONL shard.")
    parser.add_argument("--progress_every", type=int, default=100000, help="Progress logging frequency.")
    parser.add_argument("--commit_every", type=int, default=5000, help="SQLite upsert batch size.")
    parser.add_argument(
        "--no_export_tsv",
        action="store_true",
        help="Skip exporting TSV snapshots from the SQLite cache.",
    )
    return parser.parse_args()


def ensure_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS reaction_counts (
            reaction TEXT PRIMARY KEY,
            count INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS compound_counts (
            compound TEXT PRIMARY KEY,
            count INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sequence_digests (
            seq_hash TEXT PRIMARY KEY,
            sequence_length INTEGER NOT NULL,
            representative_entry_id TEXT NOT NULL,
            representative_sequence TEXT NOT NULL,
            count INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS source_counts (
            source TEXT PRIMARY KEY,
            count INTEGER NOT NULL
        );
        """
    )
    conn.commit()


def flush_counters(
    conn: sqlite3.Connection,
    reaction_counts: Counter,
    compound_counts: Counter,
    source_counts: Counter,
    sequence_rows: dict,
) -> None:
    if reaction_counts:
        conn.executemany(
            """
            INSERT INTO reaction_counts (reaction, count)
            VALUES (?, ?)
            ON CONFLICT(reaction) DO UPDATE SET
                count = reaction_counts.count + excluded.count
            """,
            reaction_counts.items(),
        )
    if compound_counts:
        conn.executemany(
            """
            INSERT INTO compound_counts (compound, count)
            VALUES (?, ?)
            ON CONFLICT(compound) DO UPDATE SET
                count = compound_counts.count + excluded.count
            """,
            compound_counts.items(),
        )
    if source_counts:
        conn.executemany(
            """
            INSERT INTO source_counts (source, count)
            VALUES (?, ?)
            ON CONFLICT(source) DO UPDATE SET
                count = source_counts.count + excluded.count
            """,
            source_counts.items(),
        )
    if sequence_rows:
        conn.executemany(
            """
            INSERT INTO sequence_digests (
                seq_hash,
                sequence_length,
                representative_entry_id,
                representative_sequence,
                count
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(seq_hash) DO UPDATE SET
                count = sequence_digests.count + excluded.count
            """,
            [
                (
                    seq_hash,
                    row["sequence_length"],
                    row["representative_entry_id"],
                    row["representative_sequence"],
                    row["count"],
                )
                for seq_hash, row in sequence_rows.items()
            ],
        )
    conn.commit()
    reaction_counts.clear()
    compound_counts.clear()
    source_counts.clear()
    sequence_rows.clear()


def export_table_to_tsv(
    conn: sqlite3.Connection,
    sql: str,
    header: list[str],
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(header) + "\n")
        for row in conn.execute(sql):
            handle.write("\t".join(str(value) for value in row) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sqlite_path = out_dir / "scan_cache.sqlite"
    conn = sqlite3.connect(sqlite_path)
    ensure_tables(conn)

    shard_writer = None
    if args.sample_n > 0:
        shard_writer = JsonlShardWriter(out_dir / "record_samples", args.records_shard_size)

    reaction_counts: Counter = Counter()
    compound_counts: Counter = Counter()
    source_counts: Counter = Counter()
    sequence_rows = {}

    scanned_records = 0
    exported_records = 0
    skipped_records = 0
    invalid_reactions = 0
    started_at = time.time()

    for entry_id, payload in iter_lmdb_json_records(args.mdb_dir):
        if args.max_records is not None and scanned_records >= args.max_records:
            break

        reaction = normalize_whitespace(str(payload.get("reaction", "")))
        sequence = str(payload.get("sequence", "")).strip()
        source = normalize_whitespace(str(payload.get("source", "unknown")))

        if not reaction or not sequence:
            skipped_records += 1
            continue

        reaction_counts[reaction] += 1
        source_counts[source] += 1

        seq_hash = sequence_sha1(sequence)
        if seq_hash not in sequence_rows:
            sequence_rows[seq_hash] = {
                "sequence_length": len(sequence),
                "representative_entry_id": entry_id,
                "representative_sequence": sequence,
                "count": 1,
            }
        else:
            sequence_rows[seq_hash]["count"] += 1

        try:
            reactants, products, _ = split_reaction_equation(reaction)
            for token in reactants + products:
                _, compound_name = parse_stoichiometry(token)
                compound_counts[normalize_compound_name(compound_name)] += 1
        except ValueError:
            invalid_reactions += 1

        if shard_writer is not None and exported_records < args.sample_n:
            shard_writer.write(
                {
                    "entry_id": entry_id,
                    "reaction": reaction,
                    "sequence": sequence,
                    "source": source,
                    "sequence_sha1": seq_hash,
                }
            )
            exported_records += 1

        scanned_records += 1
        if scanned_records % args.commit_every == 0:
            flush_counters(
                conn,
                reaction_counts=reaction_counts,
                compound_counts=compound_counts,
                source_counts=source_counts,
                sequence_rows=sequence_rows,
            )

        if scanned_records % args.progress_every == 0:
            elapsed = time.time() - started_at
            records_per_second = scanned_records / elapsed if elapsed > 0 else 0.0
            print(
                f"[scan] records={scanned_records} "
                f"exported_samples={exported_records} "
                f"invalid_reactions={invalid_reactions} "
                f"rps={records_per_second:.2f}"
            )

    flush_counters(
        conn,
        reaction_counts=reaction_counts,
        compound_counts=compound_counts,
        source_counts=source_counts,
        sequence_rows=sequence_rows,
    )
    if shard_writer is not None:
        shard_writer.close()

    summary = {
        "mdb_dir": args.mdb_dir,
        "sqlite_path": str(sqlite_path),
        "scanned_records": scanned_records,
        "exported_samples": exported_records,
        "skipped_records": skipped_records,
        "invalid_reactions": invalid_reactions,
        "unique_reactions": conn.execute("SELECT COUNT(*) FROM reaction_counts").fetchone()[0],
        "unique_compounds": conn.execute("SELECT COUNT(*) FROM compound_counts").fetchone()[0],
        "unique_sequences": conn.execute("SELECT COUNT(*) FROM sequence_digests").fetchone()[0],
        "source_counts": {
            source: count
            for source, count in conn.execute(
                "SELECT source, count FROM source_counts ORDER BY count DESC"
            )
        },
        "elapsed_seconds": round(time.time() - started_at, 2),
    }
    with (out_dir / "scan_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    if not args.no_export_tsv:
        export_table_to_tsv(
            conn,
            "SELECT reaction, count FROM reaction_counts ORDER BY count DESC",
            ["reaction", "count"],
            out_dir / "unique_reactions.tsv",
        )
        export_table_to_tsv(
            conn,
            "SELECT compound, count FROM compound_counts ORDER BY count DESC",
            ["compound", "count"],
            out_dir / "unique_compounds.tsv",
        )
        export_table_to_tsv(
            conn,
            """
            SELECT seq_hash, sequence_length, count, representative_entry_id, representative_sequence
            FROM sequence_digests
            ORDER BY count DESC
            """,
            [
                "seq_hash",
                "sequence_length",
                "count",
                "representative_entry_id",
                "representative_sequence",
            ],
            out_dir / "sequence_digests.tsv",
        )
        export_table_to_tsv(
            conn,
            "SELECT source, count FROM source_counts ORDER BY count DESC",
            ["source", "count"],
            out_dir / "source_counts.tsv",
        )

    conn.close()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
