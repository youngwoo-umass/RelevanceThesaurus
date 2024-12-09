# 1. Training Phase 1


## 1-1. Training data generation

Execute the following for i in {0..115};

> python src/data_generator2/segmented_enc/es_mmp/runner/pep5_gen.py ${i}

Output:
* output/tfrecord/mmp_pep5

## 1-2. Training 

> bash pep10.sh

* Code: src/trainer_v2/per_project/transparency/mmp/pep/runner/ts_distil.py
* Input (training data): output/tfrecord/mmp_pep5
* Output (model)
  * output/model/runs/mmp_pep10/model_20000
  * (Tensorflow 2 saved model)

# Training Phase 2
## 2-1. Inference for data Gen


> python src/trainer_v2/per_project/transparency/mmp/pep_to_tt/runner2/pre_analyze_luk.py


## 2-2. Training data generation
python src/trainer_v2/per_project/transparency/mmp/pep_to_tt/datagen3/train_gen.py confs/experiment_confs/datagen_conf_pep_tt9.yaml

Used file

	• data/bert_voca.txt
	• data/msmarco/triples.train.small.tsv
	• output/mmp/lucene_krovetz/df.pickle
	• output/mmp/lucene_krovetz/dl



## 2-2. Train

> mmp/pep_tt17.sh

Training Data: pep_tt9
	
	
Training code: src/trainer_v2/per_project/transparency/mmp/pep_to_tt/runner2/train_from_tfrecord_w_reg.py
Output model: pep_tt17





3) Inference 


rr_mtc6_pep_tt17_10000