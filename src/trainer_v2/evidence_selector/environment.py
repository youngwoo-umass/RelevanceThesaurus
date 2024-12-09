from abc import ABC, abstractmethod
from typing import Tuple, List, Iterable, Any

import numpy as np

from misc_lib import ceil_divide, tensor_to_list, TimeProfiler
from trainer_v2.chair_logging import c_log
from trainer_v2.evidence_selector.defs import RLStateTensor

from trainer_v2.evidence_selector.evidence_scoring import cross_entropy, length_loss
from utils.xml_rpc_helper import ServerProxyEx


IDS = List[int]


class ConcatMaskStrategyI(ABC):
    @abstractmethod
    def apply_mask(self, input_ids, segment_ids, action) -> np.array:
        pass

    @abstractmethod
    def get_masked_input(self, state, action) -> Tuple[IDS, IDS]:
        pass

    @abstractmethod
    def get_deletable_evidence_mask(self, input_ids, segment_ids):
        pass

    @abstractmethod
    def get_query_like_segment_mask(self, input_ids, segment_ids):
        pass


class PEInfoI:
    @abstractmethod
    def get_error(self):
        pass

    @abstractmethod
    def density(self):
        pass

    @abstractmethod
    def get_reward(self):
        pass


class PEInfo(PEInfoI):
    def __init__(
            self,
            base_pred: List[float],
            rep_pred: List[float],
            num_used: int,
            max_n_tokens: int,
            error_fn,
            tolerance=0.05, density_weight=0.05):
        self.base_pred: List[float] = base_pred
        self.rep_pred: List[float] = rep_pred
        self.num_used: int = num_used
        self.max_n_tokens: int = max_n_tokens
        self.error_fn = error_fn
        self.tolerance = tolerance
        self.density_weight = density_weight

    def get_desc_head(self):
        row = ["base_pred", "rep_pred", "get_error",
               "num_used", "max_n_tokens", "density",
               "get_reward"
               ]
        return "\t".join(row)

    def get_desc(self):
        row = [self.base_pred, self.rep_pred, self.get_error(),
               self.num_used, self.max_n_tokens, self.density(),
               self.get_reward()
               ]

        def to_str(v) -> str:
            if type(v) == float:
                return "{0:.1f}".format(v)
            elif type(v) == list:
                s = ", ".join(list(map(to_str, v)))
                return f"({s})"
            else:
                return str(v)
        s_row: List[str] = list(map(to_str, row))
        return "\t".join(s_row)

    def get_error(self):
        err = self.error_fn(np.array(self.base_pred), np.array(self.rep_pred))  # [0, inf]
        return err

    def density(self):
        return length_loss(self.num_used, self.max_n_tokens)

    def get_reward(self):
        # error reward is 10 - error
        # density reward is - 0.05 * density
        err_cap = 10
        err = self.get_error()
        err = min(err, err_cap)  # [0, 5]
        err = max(self.tolerance, err)
        combined_score = (err_cap - err) - self.density_weight * self.density()
        return combined_score


class PEPEnvironment:
    def __init__(
            self,
            pep_client,
            pe_info_factory,
            apply_mask
    ):
        self.pe_info_factory = pe_info_factory
        self.pep_client = pep_client
        self.apply_mask: ConcatMaskStrategyI = apply_mask

    def request(self, items: List[Tuple[RLStateTensor, List[int]]]) -> List[List[float]]:
        payload = []
        for state, action in items:
            state_raw = state.input_ids, state.segment_ids
            item = self.apply_mask.get_masked_input(state_raw, action)
            payload.append(item)

        ret = self.pep_client.request(list(payload))
        return ret

    def get_item_results(self, items: List[Tuple[RLStateTensor, List[int]]]) -> List[PEInfo]:
        c_log.debug("PEPClient get_item_results Entry")
        # Build dictionary of base predictions
        base_items = {}
        for sa in items:
            state, action = sa
            key = state.input_ids_hash()
            if key not in base_items:
                no_del_action = [1] * len(state.input_ids)
                base_items[key] = state, no_del_action

        bases_to_calculate = list(base_items.values())
        payload = bases_to_calculate + items
        c_log.debug("Base %d, items %d", len(bases_to_calculate), len(items))

        base_preds = {}
        c_log.debug("PEPClient request %d items", len(payload))
        outputs: List[List[float]] = self.request(payload)
        c_log.debug("PEPClient received %d items ", len(outputs))

        base_outputs = outputs[:len(bases_to_calculate)]
        item_outputs = outputs[len(bases_to_calculate):]
        assert len(item_outputs) == len(items)
        c_log.debug("PEPClient request 1")

        for sa, output in zip(bases_to_calculate, base_outputs):
            state, _ = sa
            base_preds[state.input_ids_hash()] = output
        c_log.debug("PEPClient request 2")

        pe_result_list: List[PEInfo] = []
        for sa, output in zip(items, item_outputs):
            output: List[float] = output
            state, action = sa
            base_output: List[float] = base_preds[state.input_ids_hash()]
            pe_result: PEInfo = self.pe_info_factory(base_output, output, action, state)
            pe_result_list.append(pe_result)

        assert len(pe_result_list) == len(items)

        c_log.debug("PEPClient get_item_results Done")
        return pe_result_list


def concat_two_items(payload):
    def concat_item(item1, item2):
        input_ids1, segment_ids1 = item1
        input_ids2, segment_ids2 = item2
        return input_ids1 + input_ids2, segment_ids1 + segment_ids2

    concat_items = []
    n_item_d2 = ceil_divide(len(payload), 2)
    for i in range(n_item_d2):
        if i * 2 + 1 < len(payload):
            item = concat_item(payload[i * 2], payload[i * 2 + 1])
        else:
            item = concat_item(payload[i * 2], payload[i * 2])
        concat_items.append(item)
    return concat_items


def unconcat(output, n_original):
    l_decision_list: list[Any] = []
    for item in output:
        l_decision, g_decision = item
        l_decision1 = l_decision[0]
        l_decision2 = l_decision[1]
        l_decision_list.append(l_decision1)
        if len(l_decision_list) + 1 <= n_original:
            l_decision_list.append(l_decision2)
    return l_decision_list


def pretty_input_ids(t):
    l = tensor_to_list(t)
    return [t for t in l if t != 0]


class PEPClient:
    def __init__(self, server_addr, port, ):
        self.proxy = ServerProxyEx(server_addr, port)

    def request(self, payload: List[Tuple[IDS, IDS]]) -> List[List[float]]:
        concat_payload = concat_two_items(payload)
        c_log.debug("PEPClient send ENTRY")
        response = self.proxy.send(concat_payload)
        c_log.debug("PEPClient send DONE")
        output = unconcat(response, len(payload))
        return output


class PEPClientFromPredictor:
    # It expects predict_fn to takes concatenation of two sequences
    def __init__(self, predict_fn):
        self.predict_fn = predict_fn

    def request(self, payload: List[Tuple[IDS, IDS]]) -> List[List[float]]:
        concat_payload = concat_two_items(payload)
        c_log.debug("PEPClient send ENTRY")
        response = self.predict_fn(concat_payload)
        c_log.debug("PEPClient send DONE")
        output = unconcat(response, len(payload))
        return output


class PEPClientDummy:
    def __init__(self, server_addr, port, ):
        pass
        # self.proxy = ServerProxyEx(server_addr, port)

    def request(self, payload: List[Tuple[IDS, IDS]]) -> List[List[float]]:
        output = []
        for item in payload:
            input_ids, _ = item
            output.append([0.0])
        return output