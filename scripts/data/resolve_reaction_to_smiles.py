import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.reaction_utils import (
    SPECIAL_NAME_TO_SMILES,
    build_reaction_smiles,
    cactus_smiles_query,
    canonicalize_smiles,
    load_mapping_file,
    looks_like_smiles,
    normalize_compound_name,
    parse_stoichiometry,
    pubchem_smiles_query,
    split_reaction_equation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve reaction component names to canonical SMILES and build reaction SMILES."
    )
    parser.add_argument(
        "--input_reactions",
        required=True,
        help="Path to unique_reactions.tsv or a JSONL file containing reactions.",
    )
    parser.add_argument("--out_dir", required=True, help="Output directory.")
    parser.add_argument(
        "--override_mapping",
        default=None,
        help="Optional JSON/TSV/JSONL file with manual compound->smiles overrides.",
    )
    parser.add_argument(
        "--compound_cache",
        default=None,
        help="Optional existing compound cache JSON file from a previous run.",
    )
    parser.add_argument("--max_reactions", type=int, default=None, help="Optional development cap.")
    parser.add_argument("--progress_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--request_timeout", type=float, default=10.0)
    parser.add_argument("--request_interval", type=float, default=0.1)
    parser.add_argument(
        "--cache_only",
        action="store_true",
        help="Only use overrides / cache / direct SMILES parsing, without web resolution.",
    )
    parser.add_argument(
        "--skip_cactus",
        action="store_true",
        help="Skip the NCI cactus fallback even when cache_only is false.",
    )
    return parser.parse_args()


def load_compound_cache(path: str | None) -> dict:
    if path is None:
        return {}
    cache_path = Path(path)
    if not cache_path.exists():
        raise FileNotFoundError(path)
    with cache_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {normalize_compound_name(key): value for key, value in data.items()}


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def iter_reaction_rows(path: str):
    source = Path(path)
    if source.suffix == ".jsonl" or source.name.endswith(".jsonl.gz"):
        open_fn = open
        if source.name.endswith(".gz"):
            import gzip

            open_fn = gzip.open
        with open_fn(source, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                reaction = row.get("reaction")
                if reaction:
                    yield {"reaction": reaction, "count": row.get("count")}
        return

    dialect = csv.excel_tab if source.suffix == ".tsv" else csv.excel
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, dialect=dialect)
        for row in reader:
            reaction = row.get("reaction")
            if reaction:
                count = row.get("count")
                yield {"reaction": reaction, "count": int(count) if count else None}


class CompoundResolver:
    def __init__(
        self,
        manual_mapping: dict,
        cache: dict,
        cache_only: bool,
        request_timeout: float,
        request_interval: float,
        skip_cactus: bool,
    ) -> None:
        self.manual_mapping = manual_mapping
        self.cache = cache
        self.cache_only = cache_only
        self.request_timeout = request_timeout
        self.request_interval = request_interval
        self.skip_cactus = skip_cactus
        self.stats = {
            "resolved_manual": 0,
            "resolved_cache": 0,
            "resolved_special": 0,
            "resolved_direct_smiles": 0,
            "resolved_pubchem": 0,
            "resolved_cactus": 0,
            "unresolved": 0,
        }

    def resolve(self, compound_name: str) -> dict:
        normalized_name = normalize_compound_name(compound_name)
        if normalized_name in self.cache:
            record = self.cache[normalized_name]
            source = record.get("source", "cache")
            if record.get("smiles"):
                self.stats["resolved_cache"] += 1
            else:
                self.stats["unresolved"] += 1
            return record

        if normalized_name in self.manual_mapping:
            smiles = canonicalize_smiles(self.manual_mapping[normalized_name])
            record = self._build_record(compound_name, normalized_name, smiles, "manual")
            self.stats["resolved_manual"] += 1
            return self._cache_record(record)

        if normalized_name in SPECIAL_NAME_TO_SMILES:
            smiles = canonicalize_smiles(SPECIAL_NAME_TO_SMILES[normalized_name])
            record = self._build_record(compound_name, normalized_name, smiles, "special")
            self.stats["resolved_special"] += 1
            return self._cache_record(record)

        if looks_like_smiles(compound_name):
            smiles = canonicalize_smiles(compound_name)
            record = self._build_record(
                compound_name, normalized_name, smiles, "direct_smiles"
            )
            if smiles is not None:
                self.stats["resolved_direct_smiles"] += 1
            else:
                self.stats["unresolved"] += 1
            return self._cache_record(record)

        if self.cache_only:
            record = self._build_record(compound_name, normalized_name, None, "cache_only")
            self.stats["unresolved"] += 1
            return self._cache_record(record)

        smiles = pubchem_smiles_query(compound_name, timeout=self.request_timeout)
        if smiles is not None:
            time.sleep(self.request_interval)
            record = self._build_record(compound_name, normalized_name, smiles, "pubchem")
            self.stats["resolved_pubchem"] += 1
            return self._cache_record(record)

        if not self.skip_cactus:
            smiles = cactus_smiles_query(compound_name, timeout=self.request_timeout)
            if smiles is not None:
                time.sleep(self.request_interval)
                record = self._build_record(
                    compound_name, normalized_name, smiles, "cactus"
                )
                self.stats["resolved_cactus"] += 1
                return self._cache_record(record)

        record = self._build_record(compound_name, normalized_name, None, "unresolved")
        self.stats["unresolved"] += 1
        return self._cache_record(record)

    def _build_record(
        self, compound_name: str, normalized_name: str, smiles: str | None, source: str
    ) -> dict:
        return {
            "name": compound_name,
            "normalized_name": normalized_name,
            "smiles": smiles,
            "source": source,
            "status": "resolved" if smiles else "unresolved",
        }

    def _cache_record(self, record: dict) -> dict:
        self.cache[record["normalized_name"]] = record
        return record


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manual_mapping = load_mapping_file(args.override_mapping)
    cache = load_compound_cache(args.compound_cache)
    resolver = CompoundResolver(
        manual_mapping=manual_mapping,
        cache=cache,
        cache_only=args.cache_only,
        request_timeout=args.request_timeout,
        request_interval=args.request_interval,
        skip_cactus=args.skip_cactus,
    )

    reaction_rows = []
    processed_reactions = 0
    unresolved_reactions = 0
    partial_reactions = 0
    unique_compounds_seen = set()

    for row in iter_reaction_rows(args.input_reactions):
        if args.max_reactions is not None and processed_reactions >= args.max_reactions:
            break

        reaction = row["reaction"]
        try:
            reactants, products, _ = split_reaction_equation(reaction)
        except ValueError:
            reaction_rows.append(
                {
                    "reaction": reaction,
                    "count": row.get("count"),
                    "reaction_smiles": None,
                    "status": "invalid_equation",
                    "unresolved_components": [],
                }
            )
            unresolved_reactions += 1
            processed_reactions += 1
            continue

        normalized_name_to_smiles = {}
        for token in reactants + products:
            _, compound_name = parse_stoichiometry(token)
            resolution = resolver.resolve(compound_name)
            unique_compounds_seen.add(resolution["normalized_name"])
            if resolution.get("smiles"):
                normalized_name_to_smiles[resolution["normalized_name"]] = resolution["smiles"]

        reaction_resolution = build_reaction_smiles(reaction, normalized_name_to_smiles)
        reaction_resolution["count"] = row.get("count")
        reaction_rows.append(reaction_resolution)

        if reaction_resolution["status"] == "partial":
            partial_reactions += 1
        elif reaction_resolution["status"] != "resolved":
            unresolved_reactions += 1

        processed_reactions += 1
        if processed_reactions % args.progress_every == 0:
            print(
                f"[resolve] reactions={processed_reactions} "
                f"resolved_compounds={sum(v for k, v in resolver.stats.items() if k.startswith('resolved_'))} "
                f"unresolved_compounds={resolver.stats['unresolved']}"
            )

        if processed_reactions % args.save_every == 0:
            dump_json(out_dir / "compound_cache.json", resolver.cache)

    dump_json(out_dir / "compound_cache.json", resolver.cache)

    compound_rows = sorted(
        resolver.cache.values(),
        key=lambda row: (row["status"] != "resolved", row["normalized_name"]),
    )
    with (out_dir / "compound_resolutions.jsonl").open("w", encoding="utf-8") as handle:
        for row in compound_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (out_dir / "reaction_smiles.jsonl").open("w", encoding="utf-8") as handle:
        for row in reaction_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (out_dir / "reaction_smiles.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "reaction",
                "count",
                "status",
                "reaction_smiles",
                "unresolved_components_count",
                "unresolved_components",
            ]
        )
        for row in reaction_rows:
            writer.writerow(
                [
                    row["reaction"],
                    row.get("count"),
                    row["status"],
                    row.get("reaction_smiles"),
                    len(row.get("unresolved_components", [])),
                    json.dumps(row.get("unresolved_components", []), ensure_ascii=False),
                ]
            )

    summary = {
        "input_reactions": args.input_reactions,
        "processed_reactions": processed_reactions,
        "resolved_reactions": sum(1 for row in reaction_rows if row["status"] == "resolved"),
        "partial_reactions": partial_reactions,
        "unresolved_reactions": unresolved_reactions,
        "unique_compounds_seen": len(unique_compounds_seen),
        "compound_stats": resolver.stats,
        "cache_path": str(out_dir / "compound_cache.json"),
    }
    dump_json(out_dir / "resolution_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
