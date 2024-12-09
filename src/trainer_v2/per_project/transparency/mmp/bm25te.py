
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

from adhoc.bm25 import BM25_verbose
from adhoc.kn_tokenizer import KrovetzNLTKTokenizer
from trainer_v2.chair_logging import c_log


class BM25TE:
    def __init__(
            self,
            mapping: Dict[str, Dict[str, float]],
            stemmed_groups: Dict[str, List[str]],
            bm25):
        self.relevant_terms = mapping
        self.tokenizer = None

        self.df = bm25.df
        self.N = bm25.N
        self.avdl = bm25.avdl

        self.k1 = bm25.k1
        self.k2 = bm25.k2
        self.b = bm25.b
        self.n_mapping_used = 0
        self.mapping_used_flag = 0
        self.log_mapping = Counter()
        self.stemmed_groups = stemmed_groups

    def __del__(self):
        matched_query = 0
        for k, v in self.log_mapping.items():
            if v:
                matched_query += 1

        c_log.info("%d queries used table", matched_query)
        c_log.info("%d query-document pairs used table", self.n_mapping_used)

    def score(self, query, text):
        q_terms = query.split()
        t_terms = text.split()
        q_tf = Counter(q_terms)
        t_tf = Counter(t_terms)
        self.mapping_used_flag = 0
        score = self.score_from_tfs(q_tf, t_tf)
        if self.mapping_used_flag:
            self.log_mapping[query] = 1

        return score

    def score_batch(self, qd_list):
        output = []
        for q, d in qd_list:
            s = self.score(q, d)
            output.append(s)
        return output

    def score_from_tfs(self, q_tf, t_tf):
        dl = sum(t_tf.values())
        score_sum = 0
        for q_term, q_cnt in q_tf.items():
            stem_neighbor: List[str] = self.stemmed_groups[q_term]
            translation_term_set: Dict[str, float] = self.relevant_terms[q_term]

            exact_cnt = t_tf[q_term]
            stem_cnt = 0
            for q_neighbor in stem_neighbor:
                stem_cnt += t_tf[q_neighbor]

            expansion_tf = 0
            for t, cnt in t_tf.items():
                if t in translation_term_set:
                    self.n_mapping_used += 1
                    expansion_tf += cnt * translation_term_set[t]

            if expansion_tf:
                self.mapping_used_flag = 1

            tf_sum = exact_cnt + stem_cnt + expansion_tf

            t = BM25_verbose(f=tf_sum,
                             qf=q_cnt,
                             df=self.df[q_term],
                             N=self.N,
                             dl=dl,
                             avdl=self.avdl,
                             b=self.b,
                             my_k1=self.k1,
                             my_k2=self.k2
                             )
            # if expansion_tf:
            #     c_log.debug(f"tf_sum={expansion_tf}+{raw_cnt}, adding {t} to total")

            score_sum += t
        return score_sum


