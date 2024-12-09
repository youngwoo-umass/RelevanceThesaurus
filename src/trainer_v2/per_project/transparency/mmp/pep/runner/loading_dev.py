import sys

print(1)
from taskman_client.wrapper3 import report_run3
print(2)
from trainer_v2.chair_logging import c_log
print(3)
from trainer_v2.custom_loop.dataset_factories import get_pairwise_dataset_w_score
from trainer_v2.custom_loop.definitions import ModelConfig512_1
from trainer_v2.custom_loop.eval_loop import tf_run_eval
print(4)
from trainer_v2.custom_loop.neural_network_def.ts_concat_probe import TSConcatProbe
from trainer_v2.custom_loop.neural_network_def.two_seg_alt import CombineByScoreAdd
from trainer_v2.custom_loop.per_task.pairwise_trainer import TrainerForLossReturningModel, PairwiseEvaler
print(5)
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2
from trainer_v2.custom_loop.train_loop import tf_run_train
from trainer_v2.custom_loop.trainer_if import TrainerIFBase
from trainer_v2.train_util.arg_flags import flags_parser
print(6)
