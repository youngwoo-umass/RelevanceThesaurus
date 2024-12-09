from abc import ABC, abstractmethod


class BertBasedModelIF(ABC):
    @abstractmethod
    def build_model(self, bert_params, model_config):
        pass

    @abstractmethod
    def get_keras_model(self):
        pass

    @abstractmethod
    def init_checkpoint(self, init_checkpoint):
        pass
