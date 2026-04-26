import argparse
import json
import os
import time
from datetime import timedelta

import accelerate
import biotite.structure.io as bsio
import esm
import numpy as np
import torch
from accelerate.utils import InitProcessGroupKwargs
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm


def get_pLDDT(pdb_file):
    struct = bsio.load_structure(pdb_file, extra_fields=["b_factor"])
    return struct.b_factor.mean()


@torch.no_grad()
def predict_structure(model, sequence):
    if isinstance(model, DistributedDataParallel):
        out = model.module.infer([sequence])
        pdb_out = model.module.output_to_pdb(out)[0]
    else:
        out = model.infer([sequence])
        pdb_out = model.output_to_pdb(out)[0]
    return out, pdb_out


def calculate_pae(out):
    pae = (out["aligned_confidence_probs"][0].cpu().numpy() * np.arange(64)).mean(
        -1
    ) * 31
    mask = out["atom37_atom_exists"][0, :, 1] == 1
    mask = mask.cpu()
    pae = pae[mask, :][:, mask]
    return np.mean(pae)


def inference(
    model, sequences_dict, output_path, verbose=True, save_file=True, process_index=0
):
    pLDDT_list = []
    pae_dic = {}
    for idx, (entry_id, sequence) in tqdm(
        enumerate(sequences_dict.items()),
        total=len(sequences_dict),
        disable=not verbose,
    ):
        # skip if sequence is predicted before
        output_file = f"pred_sequence_{entry_id}.pdb"
        if os.path.exists(os.path.join(output_path, output_file)):
            continue

        sequence = sequence[:1024]
        if sequence == "":
            continue
        out, pdb_out = predict_structure(model, sequence)
        # pae_list.append(calculate_pae(out))
        try:
            pae_dic[output_file] = calculate_pae(out)
        except Exception as e:
            print(
                f"Process index {process_index}: Error calculating pae for {entry_id}: {e}"
            )
            continue
        if save_file:
            print(
                f"Process index {process_index}: saving result file to {os.path.join(output_path, output_file)}"
            )
            with open(os.path.join(output_path, output_file), "w") as f:
                f.write(pdb_out)
        else:
            with open("/root/tmp.pdb", "w") as f:
                f.write(pdb_out)

        pLDDT_list.append(get_pLDDT(os.path.join(output_path, output_file)))
    return pae_dic, pLDDT_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_output_path", type=str, required=True)
    args = parser.parse_args()
    process_group_kwargs = InitProcessGroupKwargs(
        backend="nccl", timeout=timedelta(seconds=14400)
    )  # 4 hours
    accelerator = accelerate.Accelerator(kwargs_handlers=[process_group_kwargs])
    model = esm.pretrained.esmfold_v1().eval().to(accelerator.device)

    with open(os.path.join(args.test_output_path, "sequence_output.tsv"), "r") as f:
        all_content = f.readlines()[1:]
    if len(all_content[0].split("\t")) == 4:
        # in test mode
        sequences_dict = {
            content.split("\t")[0]: content.split("\t")[-1].strip()
            for content in all_content
        }
    elif len(all_content[0].split("\t")) == 2:
        # in generation mode
        sequences_dict = {
            f"generated_{idx}": content.split("\t")[-1].strip()
            for idx, content in enumerate(all_content)
        }
    else:
        raise ValueError(f"Unknown sequence format: {all_content[0]}")
    sequences_names = list(sequences_dict.keys())
    sequences_names.sort(
        key=lambda x: len(sequences_dict[x])
    )  # sort by length of sequence to make the sequences with similar length in the same process

    cur_sequences_dict = {
        i: sequences_dict[i]
        for i in sequences_names[accelerator.process_index :: accelerator.num_processes]
    }
    print(
        f"Process index {accelerator.process_index} has {len(cur_sequences_dict)} sequences"
    )

    # sanity check: make sure the sequence are only in 25 amino acids
    for sequence in tqdm(
        cur_sequences_dict.values(),
        total=len(cur_sequences_dict),
        disable=not accelerator.is_main_process,
        desc="Sanity checking sequences",
    ):
        if set(sequence) - set(list("ACDEFGHIKLMNPQRSTVWYXBUZO")):
            raise ValueError(f"Sequence {sequence} contains invalid amino acids")
    print(f"Process index {accelerator.process_index} has finished sanity checking")

    output_path = os.path.join(args.test_output_path, "esmfold_results")
    if accelerator.is_main_process:
        os.makedirs(output_path, exist_ok=True)
    accelerator.wait_for_everyone()
    start_time = time.time()
    pae_dic, pLDDT_list = inference(
        model,
        cur_sequences_dict,
        output_path,
        save_file=True,
        verbose=accelerator.is_main_process,
        process_index=accelerator.process_index,
    )
    end_time = time.time()
    print(
        f"Process index {accelerator.process_index} has finished inference in {(end_time - start_time) / 60} minutes"
    )
    accelerator.wait_for_everyone()
    pLDDT_list = accelerator.gather_for_metrics(pLDDT_list)

    # make dic to list to easier to gather
    pae_dic_list = [pae_dic]
    pae_dic_list = accelerator.gather_for_metrics(pae_dic_list)
    final_pae_dic = {}
    for pae_dic in pae_dic_list:
        final_pae_dic.update(pae_dic)
    mean_pae = np.mean(list(final_pae_dic.values()))
    accelerator.print("plddt", np.mean(pLDDT_list))
    accelerator.print("pae", np.mean(mean_pae))

    if accelerator.is_main_process:
        results_path = os.path.join(args.test_output_path, "log_metrics.json")
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                result_dic = json.load(f)
        else:
            result_dic = {}
        result_dic["ESMFold pLDDT"] = float(np.mean(pLDDT_list))
        for k, v in final_pae_dic.items():
            result_dic[f"ESMFold pae_{k}"] = float(v)

        result_dic["ESMFold pae"] = float(mean_pae)

        with open(results_path, "w") as f:
            json.dump(result_dic, f, indent=4)

    accelerator.wait_for_everyone()
