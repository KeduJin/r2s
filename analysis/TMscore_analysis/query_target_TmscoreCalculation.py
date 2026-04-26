import argparse
from pathlib import Path
import subprocess
from tqdm import tqdm
import numpy as np
import json
import biotite.structure.io as bsio
import biotite.sequence.io as bseqio
from multiprocessing import Pool
import random 

tmalign_exec = "/storage/yuanfajieLab/yuanfajie/my_project/analysis/structural_comparison/TMscore/TMalign"
GT_path = "/storage/yuanfajieLab/yuanfajie/fengyuan/Pretrain/generation/af_structures"

def GTTemplateName(UniID):
    # path = f"{GT_path}/AF-{UniID}-F1-model_v4.pdb"
    path = f"{GT_path}/{UniID}.pdb"

    # if not os.path.exists(path):
    #     print(f"{path} is not exists.")
    return path

def calculatetmscore(pdb1, pdb2):
    # now we only return query tmscore
    random_id = random.randint(1, 1000000)

    outpath = f"/storage/yuanfajieLab/yuanfajie/tmpfile/tmp_{random_id}.txt"

    tmalign_cmd = f"{tmalign_exec} {pdb1} {pdb2}> {outpath}"
    subprocess.call(tmalign_cmd, shell=True)
    # read output file and get tmscore
    content = open(outpath, 'r').readlines()
    target_line = content[13]
    tm_score = float(target_line.split()[1])
    return tm_score

def read_seq_from_pdb(file_path):
    aa_codes = {
        'ALA':'A', 'CYS':'C', 'ASP':'D', 'GLU':'E',
        'PHE':'F', 'GLY':'G', 'HIS':'H', 'LYS':'K',
        'ILE':'I', 'LEU':'L', 'MET':'M', 'ASN':'N',
        'PRO':'P', 'GLN':'Q', 'ARG':'R', 'SER':'S',
        'THR':'T', 'VAL':'V', 'TYR':'Y', 'TRP':'W'}

    seq = ''
    
    for line in open(file_path):
        if line[0:4] == "ATOM":
            columns = line.split()
            if columns[2] == "CA":
                seq = seq + aa_codes[columns[3]]
    return seq

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_output_path", type=str, required=True)
    parser.add_argument("--output_structure_type", type=str, default="esmfold_results", choices=["esmfold_results", "af3_output"])
    parser.add_argument("--generation_raw_dir", type=str, default="/storage/yuanfajieLab/yuanfajie/fengyuan/Pretrain/generation/data")
    parser.add_argument("--structure_dir", type=str, default="/storage/yuanfajieLab/yuanfajie/fengyuan/Pretrain/generation/af_structures")
    parser.add_argument("--num_processes", type=int, default=32, help="Number of processes for parallel computation")
    args = parser.parse_args()
    print("Attention!!! We use qTM to calculate the structure similarity.")

    target_path = args.test_output_path
    generation_name = Path(args.test_output_path).name.replace("generation-", "").split("_default")[0] + ".tsv"
    generation_path = Path(args.generation_raw_dir)/generation_name

    uid_list = []
    with open(generation_path, 'r') as f:
        first_line = f.readline()
        for line in f:
            query_uid, target_uid = line.strip().split("\t")[-2:]
            query_uid = query_uid.split(",")
            target_uid = target_uid.split(",")
            uid_list.append((query_uid, target_uid))

    if args.output_structure_type == "esmfold_results":
        designed_structure_path = list((Path(args.test_output_path, args.output_structure_type)).glob("*.pdb"))
        designed_structure_path = sorted(designed_structure_path, key=lambda x: int(str(x).split("_")[-1].split(".")[0]))
    elif args.output_structure_type == "af3_output":
        designed_structure_path = list(Path(args.test_output_path, args.output_structure_type).glob("*/*.cif"))
        designed_structure_path = sorted(designed_structure_path, key=lambda x: int(str(x).split("_")[-2]))
    def process_single_pair(args):
        idx, query_uid, target_uid = args
        query_pdb_path = [Path(GT_path)/f"AF-{q_uid}-F1-model_v4.cif" for q_uid in query_uid]
        target_pdb_path = [Path(GT_path)/f"AF-{t_uid}-F1-model_v4.cif" for t_uid in target_uid]
        TMscore_query_list = []
        TMscore_target_list = []
        for q_pdb_path in query_pdb_path:
            TMscore_query_list.append(calculatetmscore(designed_structure_path[idx], q_pdb_path))
        for t_pdb_path in target_pdb_path:
            TMscore_target_list.append(calculatetmscore(designed_structure_path[idx], t_pdb_path))
        
        TMscore_query = np.max(TMscore_query_list)
        TMscore_target = np.max(TMscore_target_list)
        return TMscore_query, TMscore_target

    tasks = [(idx, query_uid, target_uid) for idx, (query_uid, target_uid) in enumerate(uid_list)]

    with Pool(processes=args.num_processes) as pool:
        results = list(tqdm(pool.imap(process_single_pair, tasks), total=len(tasks), desc="calculating TMscore"))

    TMscore_query_structure_list = [r[0] for r in results]
    TMscore_target_structure_list = [r[1] for r in results]
    print(f"Mean_TMscore_query_structure_{args.output_structure_type}", np.mean(TMscore_query_structure_list))
    print(f"Mean_TMscore_target_structure_{args.output_structure_type}", np.mean(TMscore_target_structure_list))
    metrics = json.load(open(Path(args.test_output_path) / "log_metrics.json", 'r'))
    metrics[f"Mean_TMscore_query_structure_{args.output_structure_type}"] = np.mean(TMscore_query_structure_list)
    metrics[f"Mean_TMscore_target_structure_{args.output_structure_type}"] = np.mean(TMscore_target_structure_list)
    metrics[f"TMscore_query_structure_{args.output_structure_type}"] = TMscore_query_structure_list
    metrics[f"TMscore_target_structure_{args.output_structure_type}"] = TMscore_target_structure_list 
    with open(Path(args.test_output_path) / "log_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)