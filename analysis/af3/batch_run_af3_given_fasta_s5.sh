source ~/miniconda3/etc/profile.d/conda.sh
conda activate foldflow-env
cd ~/fengyuan/Pretrain

input_fasta=$1


# This scripts will run msa searching on TempCluster_yungu and return the msa results to $input_dir/msa_results
# Then run af3 prediction
cd /storage/yuanfajieLab/yuanfajie/fengyuan/alphafold3
bash scripts/cross_cluster_msa_predict_s5.sh $input_fasta
