# _DIR=$1
# echo $_DIR
# for ckpt_path in $_DIR/IntervalCheckpoints/step=*0000*; do
#     python analysis/domain_matching/run_soft_domain_matching.py \
#         --verbose \
#         --test_output_path $ckpt_path/test_plddt_filter_default_output
# done

python analysis/SeqIdentity_analysis/seq_id_calculation.py \
    --verbose \
    --test_output_path $1