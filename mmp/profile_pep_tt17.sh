# -------- Environment -----
export PYTHONPATH=src 
model_root=output/model/runs
data_root=output/tfrecord

# -------- Names ----
run_name=pep_tt17
# --------- Train Parameters ---------
# 2,420,563 instances

step=100000
config_content="{\"batch_size\": 256,
		\"train_step\": ${step},
		\"save_every_n_step\": 10000,
		\"eval_every_n_step\": 1000,
		\"learning_rate\": 1e-5,
		\"steps_per_execution\": 1

}"
config_path=data/config_a/${run_name}
echo $config_content > $config_path

train_file=${data_root}/pep_tt9/*
eval_file=${data_root}/pep_tt9_val
output_dir=${model_root}/${run_name}
python3 -u src/trainer_v2/per_project/transparency/code_release_helper/import_tracker.py \
src/trainer_v2/per_project/transparency/mmp/pep_to_tt/runner2/train_from_tfrecord_w_reg.py \
    --input_files=$train_file \
    --eval_input_files=$eval_file \
    --config_path=$config_path \
    --run_name=$run_name \
    --init_checkpoint=output/model/runs/mmp_pep10_point/model_20000 \
    --output_dir=$output_dir \
    --action=train 
	confs/experiment_confs/triples_train_alignment_cands_luk.yaml

