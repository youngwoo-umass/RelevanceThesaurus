import os

from data_generator.tokenizer_wo_tf import get_tokenizer
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_encoders import AlignCandidateExtractor
import sys
import sys
from omegaconf import OmegaConf
from table_lib import tsv_iter
import itertools

from cpath import output_path
from misc_lib import path_join


def main():
    conf_path = sys.argv[1]
    job_no = int(sys.argv[2])
    conf = OmegaConf.load(conf_path)
    bert_tokenizer = get_tokenizer()
    extractor = AlignCandidateExtractor(bert_tokenizer.basic_tokenizer.tokenize)
    line_per_job = conf.line_per_job
    raw_train_iter = tsv_iter(conf.input_file)
    st = line_per_job * job_no
    ed = st + line_per_job
    itr = itertools.islice(raw_train_iter, st, ed)
    save_path = path_join(conf.save_dir, str(job_no) + ".txt")
    extractor.pre_analyze_print_per_query(itr, save_path)


if __name__ == "__main__":
    main()