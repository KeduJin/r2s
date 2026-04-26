import json
import random

import lmdb

random.seed(123)

domain_env = lmdb.open(
    "/storage/yuanfajieLab/yuanfajie/datasets/AFDB/LMDB_uid2domain_info"
)
domain_txn = domain_env.begin()
AA_SET = "GLYHISMVNPTEQDFKRWAC"
seq_env = lmdb.open("/storage/yuanfajieLab/yuanfajie/datasets/AFDB/LMDB_seqonly")
seq_txn = seq_env.begin()
test_entry_id_path = (
    "/storage/yuanfajieLab/yuanfajie/datasets/AFDB/splits/test_repid-entryid_1k.tsv"
)
test_entry_id_list = [
    line.strip().split("\t")[1] for line in open(test_entry_id_path, "r")
]


def pos_is_in_domain(pos, domain_info):
    for domain in domain_info:
        domain_pieces = domain.split("_")
        for domain_piece in domain_pieces:
            left, right = domain_piece.split("-")
            left, right = int(left), int(right)
            if pos >= left - 1 and pos < right:
                return True
    return False


disturbed_seq_dict = {}
for entry_id in test_entry_id_list:
    domain_info = domain_txn.get(entry_id.encode())
    if domain_info is None:
        print(f"Domain info not found for {entry_id}")
        continue
    domain_info = json.loads(domain_info.decode())

    seq = seq_txn.get(entry_id.encode())
    if seq is None:
        print(f"Sequence not found for {entry_id}")
        continue
    seq = seq.decode()
    seq_list = list(seq)
    disturbed_seq_list = []
    for pos in range(len(seq_list)):
        if pos_is_in_domain(pos, domain_info):
            disturbed_seq_list.append(seq_list[pos])
        else:
            disturbed_seq_list.append(random.choice(AA_SET))
    disturbed_seq = "".join(disturbed_seq_list)
    disturbed_seq_dict[entry_id] = disturbed_seq


with open("disturbed_test_seq.fasta", "w") as f:
    for entry_id, disturbed_seq in disturbed_seq_dict.items():
        f.write(f">Disturbed_{entry_id}\n")
        f.write(f"{disturbed_seq}\n")
