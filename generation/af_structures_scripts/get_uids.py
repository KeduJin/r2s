res_list = [
    "/storage/yuanfajieLab/yuanfajie/fengyuan/Pretrain/generation/data/0209-retrieval_results-TED-35M-plddt70-sameQuery.tsv",
    "/storage/yuanfajieLab/yuanfajie/fengyuan/Pretrain/generation/data/0209-retrieval_results-TED-650M-plddt70-sameQuery.tsv"
]
uid_set = set()
for res_path in res_list:
    with open(res_path, 'r') as f:
        first_line = f.readline()
        for line in f:
            query_uid, target_uid = line.strip().split("\t")[-2:]
            query_uid = query_uid.split(",")
            target_uid = target_uid.split(",")
            for q_uid in query_uid:
                uid_set.add(q_uid)
            for t_uid in target_uid:
                uid_set.add(t_uid)
print(len(uid_set))
print(list(uid_set)[:10])

with open("uids.txt", 'w') as f:
    for uid in uid_set:
        f.write(uid + "\n")