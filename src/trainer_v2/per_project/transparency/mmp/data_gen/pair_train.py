from collections import OrderedDict

from transformers import AutoTokenizer
from typing import List, Iterable, Callable, Dict, Tuple, Set

from data_generator.create_feature import create_int_feature


def get_encode_fn_for_pair_train(max_length=256):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def encode_text_pair(query, document):
        encoded_input = tokenizer.encode_plus(
            query,
            document,
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )

        input_ids = encoded_input["input_ids"]
        token_type_ids = encoded_input["token_type_ids"]
        attention_mask = encoded_input["attention_mask"]
        return input_ids, token_type_ids

    def encode_fn(q_pos_neg: Tuple[str, str, str]):
        q, d1, d2 = q_pos_neg
        feature: OrderedDict = OrderedDict()
        input_ids1, token_type_ids1 = encode_text_pair(q, d1)
        input_ids2, token_type_ids2 = encode_text_pair(q, d2)
        feature["input_ids1"] = create_int_feature(input_ids1)
        feature["token_type_ids1"] = create_int_feature(token_type_ids1)
        feature["input_ids2"] = create_int_feature(input_ids2)
        feature["token_type_ids2"] = create_int_feature(token_type_ids2)
        return feature
    return encode_fn


def get_encode_fn_for_pointwise(use_label):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def encode_text_pair(query, document):
        encoded_input = tokenizer.encode_plus(
            query,
            document,
            padding="max_length",
            max_length=256,
            truncation=True,
        )

        input_ids = encoded_input["input_ids"]
        token_type_ids = encoded_input["token_type_ids"]
        attention_mask = encoded_input["attention_mask"]
        return input_ids, token_type_ids

    def encode_fn(qd: Tuple):
        if use_label:
            q, d, label = qd
        else:
            q, d = qd
        feature: OrderedDict = OrderedDict()
        input_ids, token_type_ids = encode_text_pair(q, d)
        feature["input_ids"] = create_int_feature(input_ids)
        feature["token_type_ids"] = create_int_feature(token_type_ids)
        if use_label:
            feature["label_ids"] = create_int_feature([label])
        return feature
    return encode_fn


def get_encode_fn_for_galign_paired():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def encode_text_pair(query, document):
        encoded_input = tokenizer.encode_plus(
            query,
            document,
            padding="max_length",
            max_length=256,
            truncation=True,
        )

        input_ids = encoded_input["input_ids"]
        token_type_ids = encoded_input["token_type_ids"]
        attention_mask = encoded_input["attention_mask"]
        return input_ids, token_type_ids

    def encode_fn(q_pos_neg: Tuple[str, str, str]):
        q, d1, d2 = q_pos_neg
        feature: OrderedDict = OrderedDict()
        input_ids1, token_type_ids1 = encode_text_pair(q, d1)
        input_ids2, token_type_ids2 = encode_text_pair(q, d2)
        feature["input_ids1"] = create_int_feature(input_ids1)
        feature["token_type_ids1"] = create_int_feature(token_type_ids1)
        feature["input_ids2"] = create_int_feature(input_ids2)
        feature["token_type_ids2"] = create_int_feature(token_type_ids2)
        return feature
    return encode_fn

