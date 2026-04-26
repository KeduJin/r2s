# set env
if [ -f .env ]; then set -a; source .env; set +a; fi
_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source /opt/conda/etc/profile.d/conda.sh
conda activate r2s
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -n "$PROJ_DIR" ]; then
  cd "$PROJ_DIR"
else
  cd "$_SCRIPT_DIR/../.."
fi

python scripts/data/split_reaction_dataset_by_mmseqs.py \
  --input_path /jinkedu/enzyme-design/ReactZyme-main/data/positive_saprot.pt \
  --train_satoken_output_path /jinkedu/enzyme-design/ReactZyme-main/data/positive_saprot_mmseqs30_train.pt \
  --val_satoken_output_path /jinkedu/enzyme-design/ReactZyme-main/data/positive_saprot_mmseqs30_val.pt \
  --train_sequence_output_path /jinkedu/enzyme-design/ReactZyme-main/data/positive_sequence_mmseqs30_train.pt \
  --val_sequence_output_path /jinkedu/enzyme-design/ReactZyme-main/data/positive_sequence_mmseqs30_val.pt \
  --stats_output_path /jinkedu/enzyme-design/ReactZyme-main/data/positive_saprot_mmseqs30_split_stats.json \
  --min_seq_id 0.3 \
  --val_ratio 0.2 \
  --seed 123
