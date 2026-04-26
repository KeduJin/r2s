# Reaction2Seq 使用说明

本文档说明如何基于 `Pretrain-TED-DPLM` 运行新的 `reaction -> sequence` 流程。默认前提是：

- 原始数据：`/jinkedu/enzyme-design/hanchenchen_data/enzyme_uniprot_ncbi_omg_gopc`
- Python 环境：`r2s`
- 训练模型：`ChemBERTa encoder + Qwen3 cross-attention decoder`

## 1. 先准备环境

```bash
conda activate r2s
cd /jinkedu/enzyme-design/Pretrain-TED-DPLM
export REACTION2SEQ_RAW_MDB="/jinkedu/enzyme-design/hanchenchen_data/enzyme_uniprot_ncbi_omg_gopc"
export REACTION2SEQ_WORKDIR="/jinkedu/enzyme-design/reaction2seq_workdir"
mkdir -p "$REACTION2SEQ_WORKDIR"
```

训练和测试配置默认读取：

```bash
export REACTION2SEQ_DATA_DIR="$REACTION2SEQ_WORKDIR/build"
```

## 2. 建议先跑一个 10 万级 dev 子集

先扫描原始 `data.mdb`，统计唯一 reaction、compound 和 sequence digest：

```bash
python scripts/data/export_reaction_records.py \
  --mdb_dir "$REACTION2SEQ_RAW_MDB" \
  --out_dir "$REACTION2SEQ_WORKDIR/scan_dev" \
  --max_records 100000 \
  --sample_n 1000 \
  --progress_every 10000 \
  --commit_every 2000
```

关键产物：

- `$REACTION2SEQ_WORKDIR/scan_dev/scan_summary.json`
- `$REACTION2SEQ_WORKDIR/scan_dev/unique_reactions.tsv`
- `$REACTION2SEQ_WORKDIR/scan_dev/unique_compounds.tsv`
- `$REACTION2SEQ_WORKDIR/scan_dev/sequence_digests.tsv`

## 3. 把 reaction 文本解析成 SMILES

如果你已经有人工维护的名称到 SMILES 映射，推荐先准备一个 JSON 文件，例如：

```json
{
  "nad(+)": "NC(=O)...",
  "nadh": "NC(=O)...",
  "coa": "CC(C)(COP(..."
}
```

然后执行：

```bash
python scripts/data/resolve_reaction_to_smiles.py \
  --input_reactions "$REACTION2SEQ_WORKDIR/scan_dev/unique_reactions.tsv" \
  --out_dir "$REACTION2SEQ_WORKDIR/resolve_dev" \
  --override_mapping "$REACTION2SEQ_WORKDIR/name_to_smiles_overrides.json" \
  --progress_every 200 \
  --save_every 100
```

如果当前机器不能联网，先用缓存和人工映射模式：

```bash
python scripts/data/resolve_reaction_to_smiles.py \
  --input_reactions "$REACTION2SEQ_WORKDIR/scan_dev/unique_reactions.tsv" \
  --out_dir "$REACTION2SEQ_WORKDIR/resolve_dev" \
  --override_mapping "$REACTION2SEQ_WORKDIR/name_to_smiles_overrides.json" \
  --cache_only
```

关键产物：

- `$REACTION2SEQ_WORKDIR/resolve_dev/reaction_smiles.tsv`
- `$REACTION2SEQ_WORKDIR/resolve_dev/reaction_smiles.jsonl`
- `$REACTION2SEQ_WORKDIR/resolve_dev/compound_cache.json`
- `$REACTION2SEQ_WORKDIR/resolve_dev/resolution_summary.json`

## 4. 构建训练所需 LMDB 和 split

先用 exact dedup 跑通 dev 流程：

```bash
python scripts/data/build_sequence_cluster_split.py \
  --mdb_dir "$REACTION2SEQ_RAW_MDB" \
  --reaction_smiles_path "$REACTION2SEQ_WORKDIR/resolve_dev/reaction_smiles.tsv" \
  --out_dir "$REACTION2SEQ_WORKDIR/build_dev" \
  --max_records 100000 \
  --mode exact \
  --allow_partial_reactions \
  --progress_every 10000 \
  --commit_every 2000
```

如果已经安装 `mmseqs2`，可以改成：

```bash
python scripts/data/build_sequence_cluster_split.py \
  --mdb_dir "$REACTION2SEQ_RAW_MDB" \
  --reaction_smiles_path "$REACTION2SEQ_WORKDIR/resolve_dev/reaction_smiles.tsv" \
  --out_dir "$REACTION2SEQ_WORKDIR/build" \
  --mode exact_then_mmseqs \
  --mmseqs_bin mmseqs \
  --mmseqs_min_seq_id 0.3 \
  --mmseqs_coverage 0.8
```

关键产物：

- `$REACTION2SEQ_WORKDIR/build[_dev]/processed/LMDB_seqonly`
- `$REACTION2SEQ_WORKDIR/build[_dev]/processed/LMDB_reaction_smiles`
- `$REACTION2SEQ_WORKDIR/build[_dev]/processed/LMDB_raw_reaction`
- `$REACTION2SEQ_WORKDIR/build[_dev]/processed/LMDB_source`
- `$REACTION2SEQ_WORKDIR/build[_dev]/processed/LMDB_train_cluster`
- `$REACTION2SEQ_WORKDIR/build[_dev]/splits/train_repid-entryid.tsv`
- `$REACTION2SEQ_WORKDIR/build[_dev]/splits/valid_repid-entryid.tsv`
- `$REACTION2SEQ_WORKDIR/build[_dev]/splits/test_repid-entryid.tsv`

训练前请把真正要训练的数据目录指向 `REACTION2SEQ_DATA_DIR`：

```bash
export REACTION2SEQ_DATA_DIR="$REACTION2SEQ_WORKDIR/build_dev"
```

## 5. 开始训练

默认训练配置：

- `configs/train/qwen-chemberta-qwen3-reaction2seq.yaml`

直接运行：

```bash
bash scripts/train/qwen-chemberta-qwen3-reaction2seq.sh
```

或者显式指定配置：

```bash
bash scripts/train/train.sh configs/train/qwen-chemberta-qwen3-reaction2seq.yaml
```

如果你想把 ChemBERTa 冻结，或修改 batch size / max steps，可以直接复制并改这个 YAML。

## 5.1 用 TensorBoard 监控训练

`reaction2seq` 的训练配置默认已经开启：

- `wandb`（当前仓库训练基类会强制走 `offline`）
- `tensorboard`

训练过程中，标量会写到当前 run 目录下的 `tblogs`，例如：

```bash
output/26Y_04M_01D_13h-qwen-chemberta-qwen3-reaction2seq/tblogs
```

你可以直接启动 TensorBoard：

```bash
bash scripts/train/launch_reaction2seq_tensorboard.sh
```

这个脚本会自动寻找最新的 `reaction2seq` 训练输出目录。也可以手动指定某个 run 目录或某个 `tblogs` 目录：

```bash
bash scripts/train/launch_reaction2seq_tensorboard.sh \
  output/26Y_04M_01D_13h-qwen-chemberta-qwen3-reaction2seq

bash scripts/train/launch_reaction2seq_tensorboard.sh \
  output/26Y_04M_01D_13h-qwen-chemberta-qwen3-reaction2seq/tblogs \
  6007
```

启动后浏览器访问：

```bash
http://127.0.0.1:6006
```

在 TensorBoard 里你会看到这类训练指标：

- `train/...`
- `val/...`
- `optimizer/group_0`
- `time_per_step`
- `epoch`

## 6. 测试和生成

默认测试配置：

- `configs/test/test_reaction2seq.yaml`

测试：

```bash
bash scripts/test/launch_reaction2seq.sh \
  configs/train/qwen-chemberta-qwen3-reaction2seq.yaml \
  /path/to/IntervalCheckpoints/step=5000
```

如果要指定 test config 和 generation config：

```bash
bash scripts/test/launch_reaction2seq.sh \
  configs/train/qwen-chemberta-qwen3-reaction2seq.yaml \
  configs/test/test_reaction2seq.yaml \
  configs/generation/default.yaml \
  /path/to/IntervalCheckpoints/step=5000
```

只生成不跑测试：

```bash
bash scripts/generate/launch_reaction2seq.sh \
  configs/train/qwen-chemberta-qwen3-reaction2seq.yaml \
  /path/to/IntervalCheckpoints/step=5000
```

`R2S` 输出的 `sequence_output.tsv` 列包括：

- `entry_id`
- `reaction_smiles`
- `raw_reaction`
- `gt_seq`
- `generated_seq`

## 7. 推荐的完整执行顺序

```bash
conda activate r2s
cd /jinkedu/enzyme-design/Pretrain-TED-DPLM
export REACTION2SEQ_RAW_MDB="/jinkedu/enzyme-design/hanchenchen_data/enzyme_uniprot_ncbi_omg_gopc"
export REACTION2SEQ_WORKDIR="/jinkedu/enzyme-design/reaction2seq_workdir"

python scripts/data/export_reaction_records.py \
  --mdb_dir "$REACTION2SEQ_RAW_MDB" \
  --out_dir "$REACTION2SEQ_WORKDIR/scan_dev" \
  --max_records 100000 \
  --sample_n 1000

python scripts/data/resolve_reaction_to_smiles.py \
  --input_reactions "$REACTION2SEQ_WORKDIR/scan_dev/unique_reactions.tsv" \
  --out_dir "$REACTION2SEQ_WORKDIR/resolve_dev" \
  --override_mapping "$REACTION2SEQ_WORKDIR/name_to_smiles_overrides.json"

python scripts/data/build_sequence_cluster_split.py \
  --mdb_dir "$REACTION2SEQ_RAW_MDB" \
  --reaction_smiles_path "$REACTION2SEQ_WORKDIR/resolve_dev/reaction_smiles.tsv" \
  --out_dir "$REACTION2SEQ_WORKDIR/build_dev" \
  --max_records 100000 \
  --mode exact \
  --allow_partial_reactions

export REACTION2SEQ_DATA_DIR="$REACTION2SEQ_WORKDIR/build_dev"

bash scripts/train/qwen-chemberta-qwen3-reaction2seq.sh
```

## 8. 当前实现的注意事项

- `resolve_reaction_to_smiles.py` 支持 `PubChem` 和 `NCI cactus` 两级在线解析，但对大型辅因子名字更推荐手工维护 `override_mapping`。
- 如果机器上没有 `mmseqs2`，`build_sequence_cluster_split.py` 在 `exact_then_mmseqs` 模式下会自动退回 `exact`。
- 当前训练配置默认使用本地完整目录 `/jinkedu/enzyme-design/DeepChem/ChemBERTa-77M-MLM`。如果你把 ChemBERTa 放在别的位置，可以直接改 `configs/train/qwen-chemberta-qwen3-reaction2seq.yaml`。
- 当前测试阶段默认主要看 `gt_seq_softlcs_score`，没有再调用原 domain 任务那套结构评估脚本。
