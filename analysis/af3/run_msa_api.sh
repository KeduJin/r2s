test_output_path=$1
# make sure test_output_path are from absolute path
test_output_path=$(realpath "$test_output_path")

if [ ! -f "$test_output_path/sequence_output.fasta" ]; then
    echo "sequence_output.fasta not found, converting tsv to fasta"
    python analysis/af3/converttsv2fasta.py --test_output_path $test_output_path
fi

start_time=$(date +%s)
# run msa searching 
# bash /storage/yuanfajieLab/yuanfajie/fengyuan/ColabFold/scripts/msa_run.sh $test_output_path/sequence_output.fasta
bash /storage/yuanfajieLab/yuanfajie/fengyuan/ColabFold/scripts/msa_run_api.sh $test_output_path/sequence_output.fasta

end_time=$(date +%s)
cost_time=$((end_time - start_time))
echo "MSA searching running time: $((cost_time / 60)) minutes $((cost_time % 60)) seconds"