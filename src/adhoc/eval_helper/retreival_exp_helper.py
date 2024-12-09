from omegaconf import OmegaConf
from pytrec_eval import RelevanceEvaluator
from typing import List, Iterable, Callable, Dict, Tuple, Set

from adhoc.adhoc_retrieval import run_retrieval
from adhoc.retriever_if import RetrieverIF
from adhoc.eval_helper.pytrec_helper import load_qrels_as_structure_from_any
from adhoc.json_run_eval_helper import save_json_qres
from misc_lib import average
from table_lib import tsv_iter
from taskman_client.task_proxy import get_task_manager_proxy
from taskman_client.wrapper3 import JobContext
from trec.ranked_list_util import json_qres_to_ranked_list
from trec.trec_parse import write_trec_ranked_list_entry
from cpath import output_path
from misc_lib import path_join


# This file should not contain dataset specific codes


def run_retrieval_eval_report_w_conf(conf, retriever: RetrieverIF, do_not_report=False):
    # Collect path and name from conf
    method = conf.method
    dataset_conf_path = conf.dataset_conf_path
    dataset_conf = OmegaConf.load(dataset_conf_path)
    run_retrieval_eval_report_w_conf_inner(retriever, dataset_conf, method, do_not_report)


def run_retrieval_eval_report_w_conf_inner(retriever, dataset_conf, method, do_not_report):
    dataset_name = dataset_conf.dataset_name
    queries_path = dataset_conf.queries_path
    metric = dataset_conf.metric
    judgment_path = dataset_conf.judgment_path
    max_doc_per_query = dataset_conf.max_doc_per_query
    queries = list(tsv_iter(queries_path))
    run_name = f"{dataset_name}_{method}"
    with JobContext(run_name):
        doc_score_d = run_retrieval(retriever, queries, max_doc_per_query)
        save_json_qres(run_name, doc_score_d)

        tr_entries = json_qres_to_ranked_list(doc_score_d, run_name)

        ranked_list_save_path = path_join(output_path, "ranked_list", f"{run_name}.txt")
        write_trec_ranked_list_entry(tr_entries, ranked_list_save_path)

        qrels: Dict[str, Dict[str, int]] = load_qrels_as_structure_from_any(judgment_path)
        try:
            evaluator = RelevanceEvaluator(qrels, {metric})
            score_per_query = evaluator.evaluate(doc_score_d)
            per_query_scores = [score_per_query[qid][metric] for qid in score_per_query]
            score = average(per_query_scores)
        except Exception:
            print(metric)
            print(doc_score_d)
            raise
        print(f"{metric}:\t{score}")
        if not do_not_report:
            proxy = get_task_manager_proxy()
            proxy.report_number(method, score, dataset_name, metric)


def run_retrieval3(retriever, dataset_conf, method, do_not_report):
    dataset_name = dataset_conf.dataset_name
    queries_path = dataset_conf.queries_path
    metric = dataset_conf.metric
    judgment_path = dataset_conf.judgment_path
    max_doc_per_query = dataset_conf.max_doc_per_query
    queries = list(tsv_iter(queries_path))
    run_name = f"{dataset_name}_{method}"
    with JobContext(run_name):
        doc_score_d = run_retrieval(retriever, queries, max_doc_per_query)
        save_json_qres(run_name, doc_score_d)

        tr_entries = json_qres_to_ranked_list(doc_score_d, run_name)

        ranked_list_save_path = path_join(output_path, "ranked_list", f"{run_name}.txt")
        write_trec_ranked_list_entry(tr_entries, ranked_list_save_path)

        qrels: Dict[str, Dict[str, int]] = load_qrels_as_structure_from_any(judgment_path)
        try:
            evaluator = RelevanceEvaluator(qrels, {metric})
            score_per_query = evaluator.evaluate(doc_score_d)
            per_query_scores = [score_per_query[qid][metric] for qid in score_per_query]
            score = average(per_query_scores)
        except Exception:
            print(metric)
            print(doc_score_d)
            raise
        print(f"{metric}:\t{score}")
        if not do_not_report:
            proxy = get_task_manager_proxy()
            proxy.report_number(method, score, dataset_name, metric)

