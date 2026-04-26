import argparse
import os

import biotite.structure.io as bsio
import esm
import torch

os.environ["TORCH_HOME"] = "/storage/yuanfajieLab/yuanfajie/.cache"
import json

import accelerate
import numpy as np
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


def inference(model, sequences_dict, output_path, verbose=True, save_file=True):
    pLDDT_list = []
    pae_dic = {}
    for idx, (entry_id, sequence) in tqdm(
        enumerate(sequences_dict.items()),
        total=len(sequences_dict),
        disable=not verbose,
    ):
        sequence = sequence[:1024]
        try:
            out, pdb_out = predict_structure(model, sequence)
        except Exception as e:
            print(f"Error predicting structure for {entry_id}: {e}")
            continue
        # pae_list.append(calculate_pae(out))
        output_file = f"gt_sequence_{entry_id}.pdb"
        pae_dic[output_file] = calculate_pae(out)
        if save_file:
            print(f"saving result file to {os.path.join(output_path, output_file)}")
            with open(os.path.join(output_path, output_file), "w") as f:
                f.write(pdb_out)
        else:
            with open("/root/tmp.pdb", "w") as f:
                f.write(pdb_out)

        pLDDT_list.append(get_pLDDT(os.path.join(output_path, output_file)))
        if idx % 100 == 0:
            accelerator.wait_for_everyone()
    return pae_dic, pLDDT_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_output_path", type=str, required=True)
    args = parser.parse_args()
    accelerator = accelerate.Accelerator()
    model = esm.pretrained.esmfold_v1().eval().to(accelerator.device)

    with open(os.path.join(args.test_output_path, "sequence_output.tsv"), "r") as f:
        all_content = f.readlines()[1:]
    sequences_dict = {
        content.split("\t")[0]: content.split("\t")[-2].strip()
        for content in all_content
    }
    sequences_names = list(sequences_dict.keys())

    cur_sequences_dict = {
        i: sequences_dict[i]
        for i in sequences_names[accelerator.process_index :: accelerator.num_processes]
    }
    print(
        f"Process index {accelerator.process_index} has {len(cur_sequences_dict)} sequences"
    )

    output_path = os.path.join(args.test_output_path, "esmfold_results_GTseqs")
    if accelerator.is_main_process:
        os.makedirs(output_path, exist_ok=True)

    accelerator.wait_for_everyone()
    pae_dic, pLDDT_list = inference(
        model,
        cur_sequences_dict,
        output_path,
        save_file=True,
        verbose=accelerator.is_main_process,
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
        results_path = os.path.join(args.test_output_path, "log_metrics_GTseqs.json")
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
