from abc import ABC, abstractmethod
from typing import Dict

import tensorflow as tf


from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.modeling_common.tf_helper import apply_gradient_warning_less
from trainer_v2.custom_loop.run_config2 import RunConfig2
from trainer_v2.custom_loop.trainer_if import TrainerIF, TrainerIFBase, EmptyEvalObject


# Purpose of ModelV2IF: to define custom init checkpoint functions
#   build_model: defines keras model
class ModelV2IF(ABC):
    @abstractmethod
    def build_model(self, run_config):
        pass

    @abstractmethod
    def get_keras_model(self) -> tf.keras.models.Model:
        pass

    @abstractmethod
    def init_checkpoint(self, model_path):
        pass

    def get_train_metrics(self):
        return {}

    def init_train_metrics(self):
        pass


class ModelV3IF(ABC):
    @abstractmethod
    def build_model(self, run_config):
        pass

    @abstractmethod
    def get_keras_model(self) -> tf.keras.models.Model:
        pass

    @abstractmethod
    def init_checkpoint(self, model_path):
        pass

    def get_train_metrics(self):
        return {}

    def get_train_metrics_for_summary(self):
        return {}

    def get_eval_metrics_for_summary(self):
        return {}

    @abstractmethod
    def get_loss_fn(self):
        pass



class TrainerCommon(TrainerIFBase):
    def __init__(self, run_config: RunConfig2,
                 inner_model: ModelV2IF):
        self.run_config = run_config
        self.eval_metrics = {}
        self.eval_metrics_factory = {}
        self.batch_size = run_config.common_run_config.batch_size
        self.inner_model = inner_model

        # These variables will be initialized by build_model()
        self.train_metrics = None
        self.model = None
        self.optimizer = None

    @abstractmethod
    def get_optimizer(self):
        pass

    def get_learning_rate(self):
        if self.run_config.train_config.learning_rate_scheduling:
            c_log.info("Use learning rate scheduling")
            decay_steps = self.run_config.train_config.train_step
            lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                self.run_config.train_config.learning_rate,
                decay_steps,
                end_learning_rate=0,
                power=1.0,
                cycle=False,
                name=None
            )
        else:
            lr_schedule = self.run_config.train_config.learning_rate
        return lr_schedule

    def build_model(self):
        if self.run_config.is_training():
            self.inner_model.build_model(self.run_config)
            self.model = self.inner_model.get_keras_model()

            self.model.summary(140)
            self.train_metrics = self.inner_model.get_train_metrics()
            self.optimizer = self.get_optimizer()
            self.model.optimizer = self.optimizer
        else:
            pass
        for k, v in self.eval_metrics_factory.items():
            self.eval_metrics[k] = v()

    def do_init_checkpoint(self, init_checkpoint):
        self.inner_model.init_checkpoint(init_checkpoint)

    def set_keras_model(self, model):
        self.model = model

    def get_keras_model(self):
        return self.model

    def train_step(self, item):
        pass

    def get_train_metrics(self) -> Dict[str, tf.keras.metrics.Metric]:
        return self.train_metrics

    def get_eval_metrics(self) -> Dict[str, tf.keras.metrics.Metric]:
        return self.eval_metrics

    def train_callback(self):
        try:
            self.model.callback({'step': self.optimizer.iterations})
        except AttributeError:
            pass

    def get_eval_object(self, eval_batches, strategy):
        eval_object = EmptyEvalObject()
        return eval_object


class PredictionTrainerCommon(TrainerCommon):
    def __init__(self, run_config: RunConfig2,
                 inner_model: ModelV2IF):
        super(PredictionTrainerCommon, self).__init__(run_config, inner_model)

    def train_step(self, item):
        model = self.get_keras_model()
        x, y = item
        with tf.GradientTape() as tape:
            prediction = model(x, training=True)
            loss = self.loss_fn(y, prediction)

        gradients = tape.gradient(loss, model.trainable_variables)
        apply_gradient_warning_less(self.optimizer, gradients, model.trainable_variables)
        return loss
