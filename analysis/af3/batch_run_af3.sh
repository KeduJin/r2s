source ~/miniconda3/etc/profile.d/conda.sh
conda activate foldflow-env
cd ~/fengyuan/Pretrain

input_dir=$1
# make sure input_dir are from absolute path
input_dir=$(realpath "$input_dir")

if [ ! -d "$input_dir" ]; then
    echo "input_dir not found"
    exit 1
fi

python analysis/af3/converttsv2fasta.py --test_output_path $input_dir

# This scripts will run msa searching on TempCluster_yungu and return the msa results to $input_dir/msa_results
# Then run af3 prediction
cd /storage/yuanfajieLab/yuanfajie/fengyuan/alphafold3
bash scripts/cross_cluster_msa_predict.sh $input_dir/sequence_output.fasta

## write the af3 results to the log_metrics.json
python analysis/af3/read_af3_results.py --test_output_path $input_dir