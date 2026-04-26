import argparse
import json
from pathlib import Path
import biotite.structure.io as bsio
from tqdm import tqdm
from shutil import copy
import os

def get_pLDDT(pdb_file):
    struct = bsio.load_structure(pdb_file, extra_fields=["b_factor"])
    return struct.b_factor.mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_output_path", type=str, required=True)
    parser.add_argument("--structure_dir", type=str, default="esmfold_results")
    parser.add_argument("--plddt_threshold", type=int, default=75)
    parser.add_argument("--pae_threshold", type=int, default=10)
    args = parser.parse_args()

    thresholded_structure_dir = Path(args.test_output_path) / (args.structure_dir + f"_thresholded_plddt{args.plddt_threshold}_pae{args.pae_threshold}")
    os.makedirs(str(thresholded_structure_dir), exist_ok=True)

    log_metrics_file = Path(args.test_output_path)/"log_metrics.json"
    with open(log_metrics_file, "r") as f:
        log_metrics = json.load(f)

    path2metrics = {}
    structure_dir = Path(args.test_output_path)/args.structure_dir
    for pdb_file in tqdm(structure_dir.glob("*.pdb"), total=len(list(structure_dir.glob("*.pdb")))):
        plddt = get_pLDDT(str(pdb_file))
        pae = log_metrics[f"ESMFold pae_{pdb_file.name}"]
        path2metrics[pdb_file] = {
            "plddt": plddt,
            "pae": pae
        }
    
    good_count = 0
    for pdb_file, metrics in path2metrics.items():
        if metrics["plddt"] > args.plddt_threshold and metrics["pae"] < args.pae_threshold:
            copy(str(pdb_file), str(thresholded_structure_dir/Path(pdb_file).name))
            good_count += 1
    print(f"filtered {good_count} out of {len(path2metrics)} structures")