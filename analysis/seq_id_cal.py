from Bio import Align

def calculate_blast_identity(seq1, seq2):
    # 1. 创建比对器
    aligner = Align.PairwiseAligner()
    
    # 2. 设置为全局比对（类似于 BLAST 的双序列比对逻辑）
    # BLAST 默认会有特定的得分矩阵，这里使用默认匹配分=1, 不匹配=0, 罚分由 aligner 自动处理
    alignments = aligner.align(seq1, seq2)
    
    # 3. 获取得分最高的比对结果
    best_alignment = alignments[0]
    
    # 4. 提取比对信息
    # counts 属性返回 (matches, mismatches, target_gaps, query_gaps)
    counts = best_alignment.counts()
    matches = counts.identities
    # BLAST Identity 分母 = 匹配 + 错配 + 所有空位
    alignment_length = counts.aligned + counts.gaps
    
    identity_pct = (matches / alignment_length)
    
    return identity_pct

name2seq = {}
with open("/storage/yuanfajieLab/yuanfajie/fengyuan/Pretrain/IDH_seq_20-20260106.txt", 'r') as f:
    for line in f:
        if ">" in line:
            name = line.strip()
        seq = f.readline().strip()
        name2seq[name] = seq
print(len(name2seq))
target_seq = "MKGFAMLGINKLGWIEKERPVAGPYDAIVRPLAVSPCTSDIHTVFEGALGDRKNMILGHEAVGEVVEVGSEVKDFKPGDRVIVPCTTPDWRSLEVQAGFQQHSNGMLAGWKFSNFKDGVFGEYFHVNDADMNLAILPKDMPLENAVMITDMMTTGFHGAELADIQMGSSVVVIGIGAVGLMAIAGAKLRGAGRIIAVGSRPICVEAAKFYGATDILNYKNGDIVDQVMKLTNGKGVDRVIMAGGGSETLEQAVRMVKPGGIISNINYHGSGDALLIPRVEWGCGMAHKTIKGGLCPGGRLRAEMLRDMVVYNRVDLSKLVTHVYHGFDHIEEALLLMKDKPKDLIKAVVIL"
target_seq_name = "CbADH"

name2seqid = {}
for name, seq in name2seq.items():
    name2seqid[name] = calculate_blast_identity(seq, target_seq)
# print(name2seqid)
with open("/storage/yuanfajieLab/yuanfajie/fengyuan/Pretrain/IDH-seq20-vsCbADH-seqid.tsv", 'w') as f:
    f.write("name\tseqid\n")
    for name, seqid in name2seqid.items():
        f.write(f"{name}\t{seqid}\n")
# # 测试
# seq_a = "ATGCGTACGT"
# seq_b = "ATGCCGACGT"
# print(calculate_blast_identity(seq_a, seq_b))