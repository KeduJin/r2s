# read the af3 metrics, e.g. mean plddt

import argparse
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_output_path", type=str, required=True)
    args = parser.parse_args()

    # check if the test_output_path is a valid path
    af_dir = Path(args.test_output_path)/"af3_output"
    if not af_dir.exists():
        raise ValueError(f"The af3 directory {af_dir} does not exist")

    # read the af3 results
    ## plddt
    plddt_list = []
    metrics_json_list = list(af_dir.glob("*/*confidences.json"))
    metrics_json_list = [i for i in metrics_json_list if not str(i).endswith("summary_confidences.json")]
    assert len(metrics_json_list) == 1000, f"There should be only 1000 metrics.json files, but got {len(metrics_json_list)}"
    for metrics_json in tqdm(metrics_json_list):
        with open(metrics_json, "r") as f:
            metrics = json.load(f)
        plddt_list.append(np.mean(metrics["atom_plddts"]))
    
    ## ptm 
    ptm_list = []
    metrics_json_list = list(af_dir.glob("*/*summary_confidences.json"))
    assert len(metrics_json_list) == 1000, f"There should be only 1000 summary_confidences.json files, but got {len(metrics_json_list)}"
    for metrics_json in tqdm(metrics_json_list):
        with open(metrics_json, "r") as f:
            metrics = json.load(f)
        ptm_list.append(metrics["ptm"])
    
    ## write into json file 
    target_json_file = Path(args.test_output_path)/"log_metrics.json"
    target_json_file_content = json.load(open(target_json_file, "r"))
    target_json_file_content["AF3 pLDDT"] = np.mean(plddt_list)
    target_json_file_content["AF3 pTM"] = np.mean(ptm_list)
    print(f"{args.test_output_path}: AF3 pLDDT: {np.mean(plddt_list)}")
    print(f"{args.test_output_path}: AF3 pTM: {np.mean(ptm_list)}")
    with open(target_json_file, "w") as f:
        json.dump(target_json_file_content, f, indent=4)