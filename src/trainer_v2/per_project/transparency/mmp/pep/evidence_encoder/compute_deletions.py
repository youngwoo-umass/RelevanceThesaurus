import pickle

import numpy as np
import os
import sys
from typing import List, Iterable, Callable, Dict, Tuple, Set, NamedTuple
import random

from data_generator.tokenizer_wo_tf import pretty_tokens
from list_lib import pairzip, lflatten, lmap
from port_info import LOCAL_DECISION_PORT

from transformers import AutoTokenizer

from data_generator2.segmented_enc.es_common.es_two_seg_common import Segment1PartitionedPair, BothSegPartitionedPair, \
    IndicesPartitionedSegment, RangePartitionedSegment
from data_generator2.segmented_enc.es_common.partitioned_encoder import apply_segmentation_to_seg1, \
    get_both_seg_partitioned_to_input_ids2
from data_generator2.segmented_enc.es_mmp.pep_attn_common import iter_attention_data_pair
from data_generator2.segmented_enc.es_nli.evidence_candidate_gen import pool_delete_indices
from misc_lib import ceil_divide, two_digit_float, path_join
from tab_print import print_table
from trainer.promise import PromiseKeeper, MyFuture
from trainer_v2.chair_logging import c_log
from trainer_v2.evidence_selector.environment import PEPClient


class PartialSequence(NamedTuple):
    delete_indices: List[int]
    tokens: List[str]


class RandomSeqDeleter:
    def __init__(self, g):
        self.g = g

    def get_delete_indices(self, segment_2_len):
        g = self.g

        g_inv = int(1 / g)
        max_del = ceil_divide(segment_2_len, g_inv)
        num_del = random.randint(1, max_del)
        raw_indices: list[int] = pool_delete_indices(num_del, segment_2_len, g)
        indices = list(set(raw_indices))
        indices.sort()
        return indices


def get_delete_indices_n_gram_selection(segment_2_len, select_size_max):
    i_start = random.randint(0, segment_2_len - select_size_max)
    i_end = min(i_start + select_size_max, segment_2_len)
    indices = list(range(i_start)) + list(range(i_end, segment_2_len))
    return indices


def delete_compute_pep(deleter, pk: PromiseKeeper, tokenizer, pair_data):
    segment2_tokens: list[str] = tokenizer.tokenize(pair_data.segment2)
    pair: Segment1PartitionedPair = apply_segmentation_to_seg1(tokenizer, pair_data)
    segment_2_len = len(segment2_tokens)
    n_sample = 10
    segment1: RangePartitionedSegment = pair.segment1
    seg1_part1_len = len(segment1.tokens) - (segment1.ed - segment1.st)
    seg1_part2_len = (segment1.ed - segment1.st)
    c_log.debug(f"Query has {seg1_part1_len} / {seg1_part2_len} tokens", )
    c_log.debug("Label: ", pair_data.label)
    indices_list = [deleter.get_delete_indices(segment_2_len) for _ in range(n_sample)]
    # Adding short evidence case using list comprehension.
    if seg1_part2_len < 6:
        for _ in range(n_sample):
            indices = get_delete_indices_n_gram_selection(segment_2_len, select_size_max=seg1_part2_len)
            indices_list.append(indices)

    def get_both_seg_partitioned_pair(indices):
        segment2 = IndicesPartitionedSegment(segment2_tokens, indices, indices)
        c = BothSegPartitionedPair(segment1, segment2, pair_data)
        return c

    candidate_list: List[BothSegPartitionedPair] = lmap(get_both_seg_partitioned_pair, indices_list)
    future_score_list = [pk.get_future(c) for c in candidate_list]
    save_item_per_qd: tuple[list[BothSegPartitionedPair], list[MyFuture[np.array]]] = candidate_list, future_score_list

    # def seg_to_text(seg):
    #     if type(seg) == tuple:
    #         s1, s2 = seg
    #         return pretty_tokens(s1, True) + " [MASK] " + pretty_tokens(s2, True)
    #     else:
    #         return pretty_tokens(seg, True)
    #
    # for part_no in [0, 1]:
    #     seg1_part = segment1.get(part_no)
    #     c_log.debug(f"Query Part %d: %s", part_no + 1, seg_to_text(seg1_part))
    return save_item_per_qd, future_score_list


def unpack_future(item):
    if isinstance(item, MyFuture):
        return item.get()

    if isinstance(item, tuple):
        return tuple([unpack_future(sub_item) for sub_item in item])
    if isinstance(item, list):
        return [unpack_future(sub_item) for sub_item in item]
    else:
        c_log.warning("Type {} is not known return as is".format(type(item)))
        return item





def main():
    # Enum qt, D
    #   Sample evidences from D
    #
    save_dir = sys.argv[1]
    part_no = int(sys.argv[2])
    server = "localhost"
    if "PEP_SERVER" in os.environ:
        server = os.environ["PEP_SERVER"]
    c_log.info("PEP_SERVER: {}".format(server))

    max_seq_length = 256
    pep_client = PEPClient(server, LOCAL_DECISION_PORT)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    EncoderType = Callable[[BothSegPartitionedPair], Iterable[tuple[list, list]]]
    encoder_iter: EncoderType = get_both_seg_partitioned_to_input_ids2(tokenizer, max_seq_length)

    def pep_client_request(candidate_itr: Iterable[BothSegPartitionedPair]) -> np.array:
        input_seg_ids: list[tuple[list, list]] = lflatten(map(encoder_iter, candidate_itr))
        raw_ret: list[list[float]] = pep_client.request(input_seg_ids)
        scores_np = np.array(raw_ret)
        scores_np = np.reshape(scores_np, [-1, 2])
        return scores_np

    g = 0.5
    deleter = RandomSeqDeleter(g)
    save_batch_size = 10

    pk = PromiseKeeper(pep_client_request)
    itr = iter_attention_data_pair(part_no)
    save_items = []
    batch_idx = 0
    for pair_data, _attn in itr:
        save_item_per_qd_future = delete_compute_pep(deleter, pep_client_request, tokenizer, pair_data)
        save_items.append(save_item_per_qd_future)
        if len(save_items) > save_batch_size:
            pk.do_duty()
            save_items = unpack_future(save_items)
            save_path = path_join(save_dir, f"{part_no}_{batch_idx}")
            pickle.dump(save_items, open(save_path, "wb"))
            batch_idx += 1
            save_items = []


if __name__ == "__main__":
    main()
