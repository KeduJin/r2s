import csv
import gzip
import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

import lmdb


REACTION_ARROW_CANDIDATES = ("<=>", "<->", "=>", "->", "=")
PLUS_SPLIT_RE = re.compile(r"\s+\+\s+")
SPACE_RE = re.compile(r"\s+")
STOICHIOMETRY_RE = re.compile(
    r"^\s*(?P<count>(?:\d+(?:\.\d+)?|\d+/\d+))\s+(?P<name>.+?)\s*$"
)

# A tiny curated set of species that commonly appear in reaction equations.
# Larger cofactors are expected to be resolved from cache or an external resolver.
SPECIAL_NAME_TO_SMILES = {
    "h(+)": "[H+]",
    "h+": "[H+]",
    "proton": "[H+]",
    "water": "O",
    "h2o": "O",
    "oxygen": "O=O",
    "o2": "O=O",
    "hydrogen peroxide": "OO",
    "h2o2": "OO",
    "carbon dioxide": "O=C=O",
    "co2": "O=C=O",
    "carbon monoxide": "[C-]#[O+]",
    "co": "[C-]#[O+]",
    "ammonia": "N",
    "nh3": "N",
    "ammonium": "[NH4+]",
    "nh4(+)": "[NH4+]",
    "phosphate": "O=P([O-])([O-])[O-]",
    "orthophosphate": "O=P([O-])([O-])[O-]",
    "pyrophosphate": "O=P([O-])([O-])OP(=O)([O-])[O-]",
    "diphosphate": "O=P([O-])([O-])OP(=O)([O-])[O-]",
    "sulfate": "O=S([O-])(=O)[O-]",
    "so4(2-)": "O=S([O-])(=O)[O-]",
}


def normalize_whitespace(text: str) -> str:
    return SPACE_RE.sub(" ", text.strip())


def normalize_compound_name(name: str) -> str:
    normalized = normalize_whitespace(name)
    normalized = normalized.replace("−", "-")
    normalized = normalized.replace("–", "-")
    return normalized.lower()


def sequence_sha1(sequence: str) -> str:
    return hashlib.sha1(sequence.encode("utf-8")).hexdigest()


def open_lmdb_readonly(path: str) -> lmdb.Environment:
    return lmdb.open(
        path,
        readonly=True,
        lock=False,
        subdir=True,
        readahead=False,
        map_size=1024**4,
    )


def iter_lmdb_json_records(path: str) -> Iterator[tuple[str, dict]]:
    env = open_lmdb_readonly(path)
    try:
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                record = json.loads(value.decode("utf-8"))
                # Some LMDB datasets store metadata entries like `length` as scalars.
                if not isinstance(record, dict):
                    continue
                yield key.decode("utf-8"), record
    finally:
        env.close()


def detect_reaction_arrow(reaction: str) -> str:
    normalized = normalize_whitespace(reaction)
    for arrow in REACTION_ARROW_CANDIDATES:
        if arrow in normalized:
            return arrow
    raise ValueError(f"Unsupported reaction arrow in reaction: {reaction}")


def parse_reaction_side(side: str) -> list[str]:
    if not side.strip():
        return []
    tokens = [normalize_whitespace(token) for token in PLUS_SPLIT_RE.split(side) if token]
    return [token for token in tokens if token]


def split_reaction_equation(reaction: str) -> tuple[list[str], list[str], str]:
    normalized = normalize_whitespace(reaction)
    arrow = detect_reaction_arrow(normalized)
    left, right = normalized.split(arrow, 1)
    return parse_reaction_side(left), parse_reaction_side(right), arrow


def parse_stoichiometry(token: str) -> tuple[Optional[str], str]:
    match = STOICHIOMETRY_RE.match(token)
    if match is None:
        return None, normalize_whitespace(token)
    return match.group("count"), normalize_whitespace(match.group("name"))


def stoichiometry_to_multiplier(count: Optional[str]) -> int:
    if count is None:
        return 1
    if "/" in count:
        return 1
    try:
        value = float(count)
    except ValueError:
        return 1
    if value.is_integer() and 1 <= int(value) <= 8:
        return int(value)
    return 1


def looks_like_smiles(text: str) -> bool:
    if not text or " " in text:
        return False
    if any(ch in "[]=#@/\\" for ch in text):
        return True
    smiles_token_re = re.compile(
        r"^(?:Br|Cl|Si|Na|Ca|Li|Mg|Zn|[BCFHIKNOPSVYbcnops]|\d|"
        r"\(|\)|\[|\]|\+|-|\.|@|%[0-9]{2})+$"
    )
    return smiles_token_re.fullmatch(text) is not None


def canonicalize_smiles(smiles: str) -> Optional[str]:
    if not smiles:
        return None
    try:
        from rdkit import Chem, RDLogger
    except ImportError:
        return smiles
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def load_mapping_file(path: Optional[str]) -> Dict[str, str]:
    if path is None:
        return {}
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(path)

    if source.suffix == ".json":
        with source.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return {normalize_compound_name(key): value for key, value in data.items()}

    mapping: Dict[str, str] = {}
    open_fn = gzip.open if source.suffix.endswith("gz") else open
    with open_fn(source, "rt", encoding="utf-8", newline="") as handle:
        if source.suffix in {".jsonl", ".gz"} or source.name.endswith(".jsonl.gz"):
            for line in handle:
                row = json.loads(line)
                key = row.get("name") or row.get("compound") or row.get("reaction")
                value = row.get("smiles") or row.get("reaction_smiles")
                if key and value:
                    mapping[normalize_compound_name(key)] = value
            return mapping

        dialect = csv.excel_tab if source.suffix == ".tsv" else csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        for row in reader:
            key = row.get("name") or row.get("compound") or row.get("reaction")
            value = row.get("smiles") or row.get("reaction_smiles")
            if key and value:
                mapping[normalize_compound_name(key)] = value
    return mapping


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    open_fn = gzip.open if destination.suffix.endswith("gz") else open
    with open_fn(destination, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def pubchem_smiles_query(name: str, timeout: float = 10.0) -> Optional[str]:
    quoted = urllib.parse.quote(name)
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{quoted}/property/CanonicalSMILES/JSON"
    )
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    properties = payload.get("PropertyTable", {}).get("Properties", [])
    if not properties:
        return None
    smiles = properties[0].get("CanonicalSMILES")
    return canonicalize_smiles(smiles)


def cactus_smiles_query(name: str, timeout: float = 10.0) -> Optional[str]:
    quoted = urllib.parse.quote(name)
    url = f"https://cactus.nci.nih.gov/chemical/structure/{quoted}/smiles"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = response.read().decode("utf-8").strip()
    except Exception:
        return None
    if not payload or payload.lower().startswith("404"):
        return None
    return canonicalize_smiles(payload)


def build_reaction_smiles(
    reaction: str,
    name_to_smiles: Dict[str, str],
    special_name_to_smiles: Optional[Dict[str, str]] = None,
) -> dict:
    special_map = special_name_to_smiles or SPECIAL_NAME_TO_SMILES
    reactant_tokens, product_tokens, arrow = split_reaction_equation(reaction)
    unresolved = []

    def resolve_side(tokens: list[str]) -> list[str]:
        side_smiles = []
        for token in tokens:
            count, raw_name = parse_stoichiometry(token)
            normalized_name = normalize_compound_name(raw_name)
            smiles = name_to_smiles.get(normalized_name)
            if smiles is None and normalized_name in special_map:
                smiles = special_map[normalized_name]
            if smiles is None and looks_like_smiles(raw_name):
                smiles = canonicalize_smiles(raw_name)
            if smiles is None:
                unresolved.append(
                    {
                        "token": token,
                        "compound": raw_name,
                        "normalized_compound": normalized_name,
                        "side": "reactant" if tokens is reactant_tokens else "product",
                    }
                )
                continue
            canonical_smiles = canonicalize_smiles(smiles) or smiles
            multiplier = stoichiometry_to_multiplier(count)
            side_smiles.extend([canonical_smiles] * multiplier)
        return side_smiles

    reactant_smiles = resolve_side(reactant_tokens)
    product_smiles = resolve_side(product_tokens)

    if reactant_smiles and product_smiles and not unresolved:
        status = "resolved"
    elif reactant_smiles or product_smiles:
        status = "partial"
    else:
        status = "unresolved"

    reaction_smiles = None
    if reactant_smiles and product_smiles:
        reaction_smiles = ".".join(reactant_smiles) + ">>" + ".".join(product_smiles)

    return {
        "reaction": reaction,
        "arrow": arrow,
        "reactants": reactant_tokens,
        "products": product_tokens,
        "reactant_smiles": reactant_smiles,
        "product_smiles": product_smiles,
        "reaction_smiles": reaction_smiles,
        "unresolved_components": unresolved,
        "status": status,
    }
