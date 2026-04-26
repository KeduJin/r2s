import argparse
import os 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_output_path", type=str, required=True)
    args = parser.parse_args()

    with open(os.path.join(args.test_output_path, "sequence_output.tsv"), "r") as f:
        lines = f.readlines()
    fasta_dict = {}
    if len(lines[0].split("\t")) == 4:
        mode = "test"
    elif len(lines[0].split("\t")) == 2:
        mode = "generation"
    else:
        raise ValueError(f"Unknown sequence format: {lines[0]}")

    for idx, line in enumerate(lines[1:]):
        if mode == "test":
            entry_id, domains, gt_seq, pred_seq = line.strip().split("\t")
        elif mode == "generation":
            domains, pred_seq = line.strip().split("\t")
            entry_id = "generation"
        fasta_dict[f"entry_{entry_id}_idx_{idx}"] = pred_seq
    
    with open(os.path.join(args.test_output_path, "sequence_output.fasta"), "w") as f:
        for entry_id, seq in fasta_dict.items():
            f.write(f">{entry_id}\n{seq}\n")