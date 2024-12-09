import sys

from cpath import output_path
from misc_lib import path_join
from trainer_v2.per_project.transparency.mmp.bias.common import load_car_maker_list
from trainer_v2.per_project.transparency.mmp.bias.inference_w_keyword_swap import run_inference_inner2
from trainer_v2.per_project.transparency.mmp.bias.parse_manual_selection import parse_dict_on_lines


def main():
    model_path = sys.argv[1]
    source_path = sys.argv[2]

    term_list = load_car_maker_list()
    term_list_set = set(term_list)

    l_dict = parse_dict_on_lines(source_path)
    score_log_f = open(path_join(output_path, "mmp", "bias", "car_exp", "exp2", f"res.tsv"), "w")
    run_inference_inner2(l_dict, model_path, term_list_set, term_list, score_log_f)


if __name__ == "__main__":
    main()
