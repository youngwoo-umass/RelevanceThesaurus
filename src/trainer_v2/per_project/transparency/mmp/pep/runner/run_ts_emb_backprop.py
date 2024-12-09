import sys
from typing import Iterable, Tuple

import tensorflow as tf

from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.hf_encode_helper import combine_with_sep_cls_inner
from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.definitions import ModelConfig512_2, ModelConfig512_1
from trainer_v2.custom_loop.neural_network_def.ts_emb_backprop import TSEmbBackprop
from trainer_v2.custom_loop.neural_network_def.two_seg_alt import CombineByScoreAdd
from trainer_v2.custom_loop.per_task.pairwise_trainer import TrainerForLossReturningModel
from trainer_v2.custom_loop.run_config2 import get_run_config2_train, RunConfig2
from trainer_v2.custom_loop.train_loop import tf_run_train
from trainer_v2.custom_loop.trainer_if import TrainerIFBase
from trainer_v2.per_project.cip.tfrecord_gen import pad_to_length
from trainer_v2.per_project.transparency.mmp.pep.pep_rerank_w_es import InputIdsSegmentIds
from trainer_v2.train_util.arg_flags import flags_parser


class TrainerHere(TrainerForLossReturningModel):
    def __init__(
            self,
            run_config: RunConfig2,
            inner_model: TSEmbBackprop):
        self.emb_model: TSEmbBackprop = inner_model
        super(TrainerForLossReturningModel, self).__init__(run_config, inner_model)
        self.neg_spe_emb_idx = -1
        self.baseline_loss = None
        self.tokenizer = get_tokenizer()

    def do_init_checkpoint(self, init_checkpoint):
        pass

    def get_eval_object(self, eval_batches, strategy):
        self.eval_batches = eval_batches
        self.strategy = strategy
        return self

    def do_eval(self):
        extra_emb_layer: tf.keras.layers.Embedding = self.emb_model.get_extra_embedding_layer()
        word_emb_layer: tf.keras.layers.Embedding = self.emb_model.get_word_embedding_layer()

        def get_row_of_embedding(emb_layer, i):
            embedding_weights = emb_layer.get_weights()[0]
            ith_embedding: tf.Tensor = embedding_weights[i]
            return ith_embedding

        target_emb = get_row_of_embedding(extra_emb_layer, self.neg_spe_emb_idx)
        bias_to_subtract = get_row_of_embedding(extra_emb_layer, 0)
        effective_emb = target_emb + bias_to_subtract

        if self.baseline_loss is None:
            with self.strategy.scope():
                baseline_dataset = self.build_baseline_dataset(256, False)

            model = self.emb_model.get_keras_model()
            for batch in baseline_dataset:
                pred, loss = model.predict_on_batch(batch)
                c_log.info(f"Baseline loss={loss}")
                self.baseline_loss = loss
                break

        all_word_emb = word_emb_layer.get_weights()[0]
        print("effective_emb", effective_emb.shape)
        print("all_word_emb", all_word_emb.shape)
        sim = tf.reduce_sum(all_word_emb * effective_emb, axis=1)
        print("sim", sim.shape)
        rank = tf.argsort(sim)
        for i in range(5):
            rank_i = rank[i].numpy().tolist()
            print(f"Rank {i}:  ", rank[i], sim[rank_i], self.tokenizer.inv_vocab[rank_i])

        metrics = {}
        dummy_loss = tf.constant(0.0)
        return dummy_loss, metrics


    def build_single_item_dataset(self, max_seq_length, is_for_training, sequence):
        SpecI = tf.TensorSpec([max_seq_length], dtype=tf.int32)
        sig = (SpecI, SpecI)

        def generator() -> Iterable[Tuple[InputIdsSegmentIds]]:
            input_ids, segment_ids = sequence

            input_ids = pad_to_length(input_ids, max_seq_length)
            segment_ids = pad_to_length(segment_ids, max_seq_length)
            yield input_ids, segment_ids

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=sig)

        dataset = dataset.batch(1)
        if is_for_training:
            dataset = dataset.repeat()
        return dataset

    def build_dataset_spe1(self, max_seq_length, is_for_training, target_q_token="when"):
        tokenizer = get_tokenizer()

        def get_sequence():
            special_token = "[unused1]"
            tokens1 = ["[MASK]"] * 2 + [target_q_token] + ["[MASK]"] * 4
            tokens2 = ["[MASK]"] * 4 + [special_token] + ["[MASK]"] * 8
            tokens, segment_ids = combine_with_sep_cls_inner(max_seq_length, tokens1, tokens2)
            idx = tokens.index(special_token)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_ids[idx] = self.neg_spe_emb_idx
            return input_ids, segment_ids

        return self.build_single_item_dataset(max_seq_length, is_for_training, get_sequence())

    def build_baseline_dataset(self, max_seq_length, is_for_training, target_q_token="when"):
        tokenizer = get_tokenizer()

        def get_sequence():
            tokens1 = ["[MASK]"] * 2 + [target_q_token] + ["[MASK]"] * 4
            tokens2 = ["[MASK]"] * 4 + [target_q_token] + ["[MASK]"] * 8
            tokens, segment_ids = combine_with_sep_cls_inner(max_seq_length, tokens1, tokens2)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            return input_ids, segment_ids

        return self.build_single_item_dataset(max_seq_length, is_for_training, get_sequence())


@report_run3
def main(args):
    c_log.info(__file__)
    run_config: RunConfig2 = get_run_config2_train(args)
    run_config.print_info()

    model_config = ModelConfig512_1()
    segment_len: int = int(model_config.max_seq_length / 2)
    task_model: TSEmbBackprop = TSEmbBackprop(model_config, CombineByScoreAdd)
    trainer: TrainerHere = TrainerHere(run_config, task_model)

    def build_dataset(input_files, is_for_training):
        # Train data indicate Template for (qt, dt)
        # Eg., (when, [SPE1])
        # E.g, (when, in [SEP1])
        return trainer.build_dataset_spe1(segment_len, is_for_training)

    return tf_run_train(run_config, trainer, build_dataset)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
