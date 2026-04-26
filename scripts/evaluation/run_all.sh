# set env
if [ -f .env ]; then set -a; source .env; set +a; fi
source $CONDA_PATH
conda activate $ENV_NAME
if [ -f .env ]; then set -a; source .env; set +a; fi
cd $PROJ_DIR


OUT_PATH=$1
python analysis/gather_test_tsv.py --test_output_path $OUT_PATH
python analysis/domain_matching/run_hard_domain_matching.py --test_output_path $OUT_PATH
accelerate launch analysis/esmfold/run_ddp.py --test_output_path  $OUT_PATH
# write esmfold pLDDT and domain pLDDT and linker pLDDT
python analysis/esmfold/write_esmfold_plddt.py --test_output_path $OUT_PATH

bash scripts/evaluation/run_gt_seq_id_cal.sh $OUT_PATH
bash scripts/evaluation/run_soft_domain_matching.sh.py  $OUT_PATH
bash scripts/evaluation/run_gt_tmscore.sh $OUT_PATH
bash scripts/evaluation/run_domain_tmscore.sh $OUT_PATH