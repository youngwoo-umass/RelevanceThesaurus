import os.path
from collections import Counter
from typing import List, Iterable, Callable

from adhoc.other.bm25_retriever_helper import get_tokenize_fn2
from misc_lib import get_second, TEL
from cpath import output_path
from misc_lib import path_join
from ptorch.cross_encoder.get_ce_msmarco_mini_lm import get_ce_msmarco_mini_lm_score_fn
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.misc_common import load_tsv, save_tsv


def accumulate(expansions: Iterable[tuple[list[str], list[str]]]):
    qt_count = Counter()
    co_count = Counter()
    for orig, expanded in expansions:
        for qt in orig:
            qt_count[qt] += 1
            for w in expanded:
                co_count[qt, w] += 1

    weight_map = avg_distribute(co_count, qt_count)
    return weight_map


def avg_distribute(co_count, qt_count):
    weight_map = {}
    for (qt, w), cnt in co_count.items():
        weight = cnt / qt_count[qt]
        if qt not in weight_map:
            weight_map[qt] = []

        weight_map[qt].append((w, weight))
    return weight_map


def accumulate_w_vector_sim(
        expansions: Iterable[tuple[list[str], list[str]]],
        get_dist_batch: Callable[[list[tuple[str, str]]], list[float]]):
    c_log.info("accumulate_w_vector_sim")
    qt_count = Counter()
    co_count = Counter()
    missing_count = 0
    expansions = list(expansions)
    for orig, expanded in TEL(expansions):
        print(orig, expanded)
        all_pair = set()
        for w in expanded:
            for qt in orig:
                all_pair.add((qt, w))

        all_pair_l = list(all_pair)
        score_d = dict(zip(all_pair_l, get_dist_batch(all_pair_l)))
        qt_count.update(orig)
        for w in expanded:
            cands = []
            for qt in orig:
                try:
                    d = score_d[qt, w]
                    cands.append((qt, d))
                except KeyError:
                    pass

            cands.sort(key=get_second, reverse=True)

            if cands:
                qt = cands[0][0]
                co_count[qt, w] += 1
            else:
                missing_count += 1

    weight_map = avg_distribute(co_count, qt_count)
    return weight_map


def load_expansions(file_path_iter, tokenize_fn):
    for file_path in file_path_iter:
        if os.path.exists(file_path):
            for qid, expansions, _score, query in load_tsv(file_path):
                expanded: list[str] = expansions.split()
                orig: list[str] = tokenize_fn(query)
                yield orig, expanded


def mapping_to_table(mapping: dict[str, list[tuple[str, float]]]) -> Iterable[tuple[str, str, float]]:
    for qt, entries in mapping.items():
        for dt, score in entries:
            yield qt, dt, score


def main():
    def file_path_iter_all():
        for i in range(120):
            file_path = path_join(output_path, "msmarco", "passage", "eli_q_ex", f"{i}.txt")

    def file_path_iter_head():
        yield path_join(output_path, "msmarco", "passage", "eli_q_ex", "0_head.txt")

    acc_opt = "other"
    save_path = path_join(output_path, "mmp", "tables", "eli_0_head.txt")
    path_itr = file_path_iter_head()
    tokenize_fn = get_tokenize_fn2("lucene")
    expansions = load_expansions(path_itr, tokenize_fn)

    if acc_opt == "simple":
        weight_map = accumulate(expansions)
    else:
        c_log.info("Loading ce model")
        score_fn = get_ce_msmarco_mini_lm_score_fn()

        weight_map = accumulate_w_vector_sim(expansions, score_fn)

    out_itr = mapping_to_table(weight_map)
    save_tsv(out_itr, save_path)
    c_log.info("Saved at %s", save_path)


if __name__ == "__main__":
    main()

