test_output_path=$1
# make sure test_output_path are from absolute path
test_output_path=$(realpath "$test_output_path")

python analysis/af3/converttsv2fasta.py --test_output_path $test_output_path

start_time=$(date +%s)
bash /storage/yuanfajieLab/yuanfajie/fengyuan/alphafold3/scripts/batch_inference_multigpu.sh $test_output_path/sequence_output.fasta

end_time=$(date +%s)
cost_time=$((end_time - start_time))
echo "AF3 model running time: ${cost_time}s"