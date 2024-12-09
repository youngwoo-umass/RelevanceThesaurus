from typing import Callable, List, TypeVar

import tensorflow as tf

from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.definitions import ModelConfigType, ModelConfig2Seg
from trainer_v2.custom_loop.run_config2 import RunConfig2


def parse_file_path(input_file):
    input_files = []
    for input_pattern in input_file.split(","):
        input_files.extend(tf.io.gfile.glob(input_pattern))
    return input_files


def create_dataset_common(decode_record: Callable,
                          run_config: RunConfig2,
                          file_path: str,
                          is_training_split: bool):
    do_shuffle = is_training_split and run_config.train_config.do_shuffle
    do_repeat = is_training_split
    config = run_config.dataset_config
    batch_size = run_config.common_run_config.batch_size
    if not is_training_split:
        if run_config.common_run_config.eval_batch_size is not None:
            batch_size = run_config.common_run_config.eval_batch_size
    input_files: List[str] = parse_file_path(file_path)
    if len(input_files) > 1:
        c_log.info("{} inputs files".format(len(input_files)))
    elif len(input_files) == 0:
        c_log.error("No input files found - Maybe you dont' want this ")
        raise FileNotFoundError(input_files)

    num_parallel_reads = min(len(input_files), 4)
    dataset = tf.data.TFRecordDataset(input_files, num_parallel_reads=num_parallel_reads)
    if do_shuffle:
        dataset = dataset.shuffle(config.shuffle_buffer_size)
    if do_repeat:
        dataset = dataset.repeat()
    dataset = dataset.map(decode_record,
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def create_dataset_common_inner(
        decode_record: Callable,
        file_path: str,
        do_shuffle,
        do_repeat,
        batch_size,
        shuffle_buffer_size,
        drop_remainder
):
    input_files: List[str] = parse_file_path(file_path)
    if len(input_files) > 1:
        c_log.info("{} inputs files".format(len(input_files)))
    elif len(input_files) == 0:
        c_log.error("No input files found - Maybe you dont' want this ")
        raise FileNotFoundError(input_files)

    num_parallel_reads = min(len(input_files), 4)
    dataset = tf.data.TFRecordDataset(input_files, num_parallel_reads=num_parallel_reads)
    if do_shuffle:
        dataset = dataset.shuffle(shuffle_buffer_size)
    if do_repeat:
        dataset = dataset.repeat()
    dataset = dataset.map(decode_record,
                          num_parallel_calls=tf.data.AUTOTUNE)

    if batch_size is not None:
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def get_classification_dataset(file_path,
                               run_config: RunConfig2,
                               model_config: ModelConfigType,
                               is_for_training,
                               ) -> tf.data.Dataset:
    seq_length = model_config.max_seq_length

    def select_data_from_record(record):
        for k, v in record.items():
            record[k] = tf.cast(v, tf.int32)
        entry = (record['input_ids'], record['segment_ids']), record['label_ids']
        return entry

    def decode_record(record):
        name_to_features = {
            'input_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
            'input_mask': tf.io.FixedLenFeature([seq_length], tf.int64),
            'segment_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
            'label_ids': tf.io.FixedLenFeature([], tf.int64),
        }
        record = tf.io.parse_single_example(record, name_to_features)
        return select_data_from_record(record)

    return create_dataset_common(decode_record, run_config,
                                 file_path, is_for_training)


def get_classification_dataset_hf_to_bert_f2(
        file_path,
        run_config: RunConfig2,
        model_config: ModelConfigType,
        is_for_training,
    ) -> tf.data.Dataset:
    seq_length = model_config.max_seq_length

    def select_data_from_record(record):
        for k, v in record.items():
            record[k] = tf.cast(v, tf.int32)
        entry = (record['input_ids'], record['segment_ids']), record['label_ids']
        return entry

    def decode_record(record):
        name_to_features = {
            'input_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
            'segment_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
            'label_ids': tf.io.FixedLenFeature([], tf.int64),
        }
        record = tf.io.parse_single_example(record, name_to_features)
        return select_data_from_record(record)

    return create_dataset_common(decode_record, run_config,
                                 file_path, is_for_training)


def get_sequence_labeling_dataset(file_path,
                                  run_config: RunConfig2,
                                  model_config: ModelConfigType,
                                  is_for_training,
                                  ) -> tf.data.Dataset:
    seq_length = model_config.max_seq_length

    def select_data_from_record(record):
        for k, v in record.items():
            record[k] = tf.cast(v, tf.int32)
        entry = (record['input_ids'], record['segment_ids']), record['label_ids']
        return entry

    def decode_record(record):
        name_to_features = {
            'input_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
            'input_mask': tf.io.FixedLenFeature([seq_length], tf.int64),
            'segment_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
            'label_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
        }
        record = tf.io.parse_single_example(record, name_to_features)
        return select_data_from_record(record)

    return create_dataset_common(decode_record, run_config,
                                 file_path, is_for_training)


ModelConfig2SegT = TypeVar('ModelConfig2SegT', bound=ModelConfig2Seg)


def get_two_seg_data(file_path,
                     run_config: RunConfig2,
                     model_config: ModelConfig2SegT,
                     is_for_training,
                     ) -> tf.data.Dataset:
    seq_length_list = [model_config.max_seq_length1, model_config.max_seq_length2]

    def decode_record(record):
        name_to_features = {
            'label_ids': tf.io.FixedLenFeature([], tf.int64),
        }
        for i in range(2):
            def fixed_len_feature():
                return tf.io.FixedLenFeature([seq_length_list[i]], tf.int64)

            name_to_features[f'input_ids{i}'] = fixed_len_feature()
            name_to_features[f'input_mask{i}'] = fixed_len_feature()
            name_to_features[f'segment_ids{i}'] = fixed_len_feature()

        record = tf.io.parse_single_example(record, name_to_features)
        return reform_example(record)

    def reform_example(record):
        for k, v in record.items():
            if v.dtype == tf.int64:
                record[k] = tf.cast(v, tf.int32)
        x = record['input_ids0'], record['segment_ids0'], record['input_ids1'], record['segment_ids1']
        y = record['label_ids']
        return x, y

    return create_dataset_common(decode_record,
                                 run_config,
                                 file_path,
                                 is_for_training)


def get_pairwise_dataset(
        file_path,
        run_config: RunConfig2,
        model_config: ModelConfigType,
        is_for_training,
        add_dummy_y=True,
        segment_ids_for_token_type_ids=False,
    ) -> tf.data.Dataset:
    print("get_pairwise_dataset: segment_ids_for_token_type_ids=", segment_ids_for_token_type_ids)
    def decode_record(record):
        name_to_features = {
        }
        for i in range(2):
            def fixed_len_feature():
                return tf.io.FixedLenFeature([model_config.max_seq_length], tf.int64)
            name_to_features[f'input_ids{i+1}'] = fixed_len_feature()
            if segment_ids_for_token_type_ids:
                name_to_features[f'segment_ids{i + 1}'] = fixed_len_feature()
            else:
                name_to_features[f'token_type_ids{i+1}'] = fixed_len_feature()
        print(name_to_features)
        record = tf.io.parse_single_example(record, name_to_features)
        if add_dummy_y:
            record = reform_example(record)
        return record

    def reform_example(record):
        for k, v in record.items():
            if v.dtype == tf.int64:
                record[k] = tf.cast(v, tf.int32)
        if segment_ids_for_token_type_ids:
            x = record['input_ids1'], record['segment_ids1'], record['input_ids2'], record['segment_ids2']
        else:
            x = record['input_ids1'], record['token_type_ids1'], record['input_ids2'], record['token_type_ids2']

        return x, tf.constant(1)

    return create_dataset_common(decode_record,
                                 run_config,
                                 file_path,
                                 is_for_training)


def get_pairwise_dataset_w_score(
        file_path,
        run_config: RunConfig2,
        model_config: ModelConfigType,
        is_for_training,
        segment_ids_for_token_type_ids=False,
    ) -> tf.data.Dataset:
    def decode_record(record):
        name_to_features = {}
        for i in range(2):
            def fixed_len_feature():
                return tf.io.FixedLenFeature([model_config.max_seq_length], tf.int64)
            name_to_features[f'input_ids{i+1}'] = fixed_len_feature()
            if segment_ids_for_token_type_ids:
                name_to_features[f'segment_ids{i + 1}'] = fixed_len_feature()
            else:
                name_to_features[f'token_type_ids{i+1}'] = fixed_len_feature()
            name_to_features[f'score{i+1}'] = tf.io.FixedLenFeature([1], tf.float32)

        record = tf.io.parse_single_example(record, name_to_features)
        return record

    return create_dataset_common(decode_record,
                                 run_config,
                                 file_path,
                                 is_for_training)

def get_pointwise(
            file_path,
            run_config: RunConfig2,
            model_config: ModelConfigType,
            is_for_training,
    ) -> tf.data.Dataset:

    def decode_record(record):
        name_to_features = {}
        def fixed_len_feature():
            return tf.io.FixedLenFeature([model_config.max_seq_length], tf.int64)
        name_to_features[f'input_ids'] = fixed_len_feature()
        name_to_features[f'token_type_ids'] = fixed_len_feature()
        if is_for_training:
            name_to_features["label_ids"] = tf.io.FixedLenFeature([1], tf.int64)

        record = tf.io.parse_single_example(record, name_to_features)
        return record

    return create_dataset_common(decode_record,
                                 run_config,
                                 file_path,
                                 is_for_training)


def get_pointwise_train(
            file_path,
            run_config: RunConfig2,
            model_config: ModelConfigType,
            is_for_training,
    ) -> tf.data.Dataset:
    def decode_record(record):
        name_to_features = {}
        def fixed_len_feature():
            return tf.io.FixedLenFeature([model_config.max_seq_length], tf.int64)
        name_to_features[f'input_ids'] = fixed_len_feature()
        name_to_features[f'token_type_ids'] = fixed_len_feature()
        name_to_features["label_ids"] = tf.io.FixedLenFeature([1], tf.int64)
        record = tf.io.parse_single_example(record, name_to_features)
        return record, record["label_ids"]

    return create_dataset_common(decode_record,
                                 run_config,
                                 file_path,
                                 is_for_training)

def read_pairwise_as_pointwise(
        file_path,
        run_config: RunConfig2,
        model_config: ModelConfigType,
        is_for_training,
    ) -> tf.data.Dataset:

    def decode_record(record):
        name_to_features = {
        }
        for i in range(2):
            def fixed_len_feature():
                return tf.io.FixedLenFeature([model_config.max_seq_length], tf.int64)
            name_to_features[f'input_ids{i+1}'] = fixed_len_feature()
            name_to_features[f'token_type_ids{i+1}'] = fixed_len_feature()

        record = tf.io.parse_single_example(record, name_to_features)
        return reform_example(record)

    def reform_example(record):
        # x = record['input_ids1'], record['token_type_ids1'], record['input_ids2'], record['token_type_ids2']
        return record, tf.constant(1)

    dataset = create_dataset_common(
        decode_record,
        run_config,
        file_path,
        is_for_training)

    def concat_items(x, y):
        input_ids = tf.concat([x['input_ids1'], x['input_ids2']], axis=0)
        token_type_ids = tf.concat([x['token_type_ids1'], x['token_type_ids2']], axis=0)
        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
        }

    return dataset.map(concat_items)


def build_dataset_repeat_segs(input_files, run_config, model_config, is_for_training):
    dataset = get_classification_dataset(input_files, run_config, model_config, is_for_training)

    def repeat_record(*record):
        (input_ids, segment_ids), y = record
        return (input_ids, segment_ids, input_ids, segment_ids), y

    return dataset.map(repeat_record)


def get_qd_multi_seg_dataset(
        file_path,
        run_config: RunConfig2,
        q_seq_len,
        d_seq_len,
        n_score_size,
        is_for_training,
    ) -> tf.data.Dataset:

    def decode_record(record):
        name_to_features = {}
        name_to_features[f'q_input_ids'] = tf.io.FixedLenFeature([q_seq_len], tf.int64)
        name_to_features[f'q_segment_ids'] = tf.io.FixedLenFeature([q_seq_len], tf.int64)
        name_to_features[f'd_input_ids'] = tf.io.FixedLenFeature([d_seq_len], tf.int64)
        name_to_features[f'd_segment_ids'] = tf.io.FixedLenFeature([d_seq_len], tf.int64)
        name_to_features[f'scores'] = tf.io.FixedLenFeature([n_score_size], tf.float32)
        record = tf.io.parse_single_example(record, name_to_features)
        return record

    return create_dataset_common(decode_record,
                                 run_config,
                                 file_path,
                                 is_for_training)
