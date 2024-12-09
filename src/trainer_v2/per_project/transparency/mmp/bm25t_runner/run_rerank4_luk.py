# LuK = Lucene tokenizer Krovetz stemmed
import sys

from omegaconf import OmegaConf

from adhoc.resource.dataset_conf_helper import get_rerank_dataset_conf_path
from cpath import yconfig_dir_path
from misc_lib import path_join
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.bm25t_runner.run_rerank4 import get_bm25t_scorer_fn
from trainer_v2.per_project.transparency.mmp.eval_helper.rerank_w_conf import run_rerank_with_conf_common


def run_bm25t_luk(run_name, table_path, dataset):
    get_scorer_fn = get_bm25t_scorer_fn
    dataset_conf_path = get_rerank_dataset_conf_path(dataset)
    conf = OmegaConf.create(
        {
            "bm25conf_path": path_join(yconfig_dir_path, "bm25_resource", "lucene_krovetz.yaml"),
            "dataset_conf_path": dataset_conf_path,
            "table_path": table_path,
            "table_type": "Score",
            "method": run_name,
            "run_name": run_name
        }
    )
    run_rerank_with_conf_common(conf, get_scorer_fn)
    c_log.info("Done")


def main():
    c_log.info(__file__)
    table_path = sys.argv[1]
    run_name = sys.argv[2]
    try:
        dataset = sys.argv[3]
    except IndexError:
        dataset = "dev_c"

    with JobContext(run_name):
        run_bm25t_luk(run_name, table_path, dataset)


if __name__ == "__main__":
    main()
