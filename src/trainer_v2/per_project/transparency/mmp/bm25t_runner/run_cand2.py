from cpath import output_path
from misc_lib import path_join
from trainer_v2.per_project.transparency.mmp.bm25t.bm25t_helper import run_dev_rerank_eval_with_bm25t
from trainer_v2.per_project.transparency.mmp.table_readers import load_mapping_from_align_scores


def cand_2():
    table_path = path_join(output_path, "msmarco", "passage", "align_scores", "candidate2.tsv")
    dataset = "dev_sample1000"
    cut = 0.1
    mapping_val = 0.1
    table_name = f"cand2"
    run_name = f"bm25_{table_name}"
    mapping = load_mapping_from_align_scores(table_path, cut, mapping_val)
    run_dev_rerank_eval_with_bm25t(dataset, mapping, run_name)


def main():
    cand_2()


if __name__ == "__main__":
    main()
