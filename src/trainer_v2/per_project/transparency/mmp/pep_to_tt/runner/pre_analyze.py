import itertools
from table_lib import tsv_iter
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_encoders import AlignCandidateExtractor
import sys


def main():
    from pyserini.analysis import Analyzer, get_lucene_analyzer
    analyzer = Analyzer(get_lucene_analyzer())
    extractor = AlignCandidateExtractor(analyzer.analyze)
    num_lines = 100000
    raw_train_iter = tsv_iter(sys.argv[1])
    itr = itertools.islice(raw_train_iter, num_lines)
    extractor.pre_analyze_print_unique(itr, sys.argv[2])


if __name__ == "__main__":
    main()
