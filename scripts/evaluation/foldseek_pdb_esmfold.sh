
test_output_path=$1
test_output_path=$(readlink -f "$test_output_path")
# python scripts/evaluation/foldseek_preprocess.py \
#      --test_output_path $test_output_path\
#      --structure_dir esmfold_results\
#      --plddt_threshold 75\
#      --pae_threshold 10

cd /storage/yuanfajieLab/yuanfajie/my_project/analysis/tmscore_foldseek/workdir
source ~/miniconda3/etc/profile.d/conda.sh
conda activate foldseek
start_time=$(date +%s)

QUERY_DIR=$test_output_path/esmfold_results
OUTPUT_DIR=$test_output_path
OUTPUT_BASENAME=esmfold_results
echo "QUERY_DIR: $QUERY_DIR"
echo "OUTPUT_FILE: ${OUTPUT_DIR}/${OUTPUT_BASENAME}_vs_pdb.txt"
echo "Running foldseek easy-search..."
foldseek easy-search $QUERY_DIR pdb ${OUTPUT_DIR}/${OUTPUT_BASENAME}_vs_pdb.txt tmpFolder\
     --alignment-type 1\
     --format-output query,target,qtmscore,ttmscore,alntmscore,lddt\
     --tmscore-threshold 0.0\
     --exhaustive-search\
     --max-seqs 10000000000\
     --threads 120

end_time=$(date +%s)

duration=$((end_time - start_time))
echo "Elpased time: $duration s"