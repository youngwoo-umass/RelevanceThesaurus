# -------- Environment -----
export PYTHONPATH=src 
model_root=output/model/runs
data_root=output/tfrecord

# -------- Names ----
run_name=mmp_pep10
# --------- Train Parameters ---------
# 2,420,563 instances

step=20000
config_content="{\"batch_size\": 16,
		\"train_step\": ${step},
		\"save_every_n_step\": 2000,
		\"eval_every_n_step\": ${step},
		\"learning_rate_scheduling\": \"linear\",
		\"steps_per_execution\": 10

}"
config_path=data/config_a/${run_name}
echo $config_content > $config_path

train_file=${data_root}/mmp_pep5/*
eval_file=${data_root}/mmp_pep5/0
output_dir=${model_root}/${run_name}

python3 -u src/trainer_v2/per_project/transparency/mmp/pep/runner/ts_score_add.py \
    --input_files=$train_file \
    --eval_input_files=$eval_file \
    --config_path=$config_path \
    --run_name=$run_name \
    --init_checkpoint=output/model/runs/uncased_L-12_H-768_A-12/bert_model.ckpt \
    --output_dir=$output_dir \
    --action=train 

if [ "$?" -ne "0" ];then
       echo "training failed"	
       exit 1
fi


