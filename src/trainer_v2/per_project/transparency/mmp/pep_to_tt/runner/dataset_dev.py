import sys
import time

import tensorflow as tf
from omegaconf import OmegaConf

from misc_lib import get_dir_files
from table_lib import tsv_iter
from trainer_v2.custom_loop.run_config2 import RunConfig2
from trainer_v2.per_project.transparency.mmp.pep_to_tt.dataset_builder import PEP_TT_EncoderMulti
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_modeling import PEP_TT_ModelConfig
from trainer_v2.per_project.transparency.mmp.pep_to_tt.runner.run_train_pep_tt import get_run_config


def main():
    conf = OmegaConf.load(sys.argv[1])
    run_config: RunConfig2 = get_run_config(conf)
    model_config = PEP_TT_ModelConfig()
    file_list = get_dir_files(conf.train_data_dir)
    is_training = True
    tf.io.decode_csv(
        records,
        record_defaults,
        field_delim=',',
        use_quote_delim=True,
        na_value='',
        select_cols=None,
        name=None
    )

    encoder = PEP_TT_EncoderMulti(model_config, conf)
    def process_row(line):
        q, d_pos, d_neg = line.split("\t")
        feature_d = encoder.encode_triplet(q, d_pos, d_neg)
        return feature_d

    dataset = tf.data.TextLineDataset(file_list)
    num_parallel_reads = min(4, len(file_list))
    dataset = dataset.interleave(
        process_row,
        num_parallel_calls=tf.data.AUTOTUNE,
        cycle_length=1
    )
    dataset = dataset.batch(32, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    last = time.time()
    for item in dataset:
        now = time.time()
        print(now - last)
        last = now


if __name__ == "__main__":
    main()