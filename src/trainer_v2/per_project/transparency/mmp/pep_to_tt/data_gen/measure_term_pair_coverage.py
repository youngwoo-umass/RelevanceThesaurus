import json
import math
import random
from typing import Iterable

from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.msmarco.passage.path_helper import get_train_triples_partition_path
from misc_lib import pick1, ceil_divide, path_join
from table_lib import tsv_iter
from trainer_v2.per_project.transparency.mmp.pep.seg_enum_helper import TextRep



def main():
    job_no = 0
    file_path = get_train_triples_partition_path(job_no)
    random.seed(0)
    tokenizer = get_tokenizer()
    raw_train_iter: Iterable[tuple[str, str, str]] = tsv_iter(file_path)
    voca: list[str] = NotImplemented
    voca_d: dict[str, int] = NotImplemented
    save_dir: str = NotImplemented

    term_for_worker = 1000
    n_worker = ceil_divide(len(voca), term_for_worker)

    f_arr = []
    for i in range(n_worker):
        f = open(path_join(save_dir, str(i)), "w")
        f_arr.append(f)

    def get_out_f(q_term):
        term_idx = voca_d[q_term]
        worker_i = term_idx // term_for_worker
        return f_arr[worker_i]


    # TODO Check why file no is like this / Check if triplet iter is better than mine
    def get_match_candidate(q, d):
        q = TextRep.from_text(tokenizer, q)
        d: TextRep = TextRep.from_text(tokenizer, d)

        d_term_list = list(d.counter.keys())
        q_term_cand = []
        for q_term, qtf, _ in q.get_bow():
            exact_match_cnt: int = d.counter[q_term]
            if not exact_match_cnt:
                q_term_cand.append(q_term)

        q_term_cand = [t for t in q_term_cand if t in voca_d]
        q_term = pick1(q_term_cand)
        return q_term, d_term_list

    for idx, (q, d_pos, d_neg) in enumerate(raw_train_iter):
        data_id_pos = idx * 10 + 1
        data_id_neg = idx * 10 + 2
        for q, d, data_id in [(q, d_pos, data_id_pos), (q, d_neg, data_id_neg)]:
            q_term, d_term_list = get_match_candidate(q, d)
            q_term_id = voca_d[q_term]
            d_term_ids = [voca_d[t] for t in d_term_list]
            out_row = [data_id, q_term_id, d_term_ids]
            out_f = get_out_f(q_term)
            s = json.dumps(out_row)
            out_f.write(s + "\n")






if __name__ == "__main__":
    main()




