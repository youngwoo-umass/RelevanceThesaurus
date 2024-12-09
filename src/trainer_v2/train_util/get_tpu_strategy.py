import tensorflow as tf

from taskman_client.task_proxy import get_local_machine_name, assign_tpu_anonymous
from trainer_v2.chair_logging import c_log
import atexit


def device_list_summary(device_list):
    if not device_list:
        return "No device found"
    n_gpu = 0
    name_set = set()
    for dev in device_list:
        if dev.device_type == 'GPU':
            if dev.name in name_set:
                c_log.warn("Duplicated name {}".format(dev.name))
            name_set.add(dev.name)
            n_gpu += 1
    if n_gpu == len(device_list):
        return "{} GPUs found".format(n_gpu)
    else:
        return str(device_list)


def get_strategy(use_tpu=False, tpu_name=None):
    if use_tpu:
        strategy = get_tpu_strategy_inner(tpu_name)
    else:
        c_log.debug("use_tpu={}".format(use_tpu))
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        c_log.info(device_list_summary(tf.config.list_logical_devices('GPU')))
        try:
            atexit.register(strategy._extended._cross_device_ops._pool.close)  # type: ignore
            atexit.register(strategy._extended._host_cross_device_ops._pool.close)  # type: ignore
        except AttributeError:
            c_log.warning("Skip atexit.register")
    return strategy


def get_strategy2(use_tpu, tpu_name=None, force_use_gpu=False):
    if use_tpu:
        strategy = get_tpu_strategy_inner(tpu_name)
    else:
        c_log.debug("use_tpu={}".format(use_tpu))
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        gpu_devices = tf.config.list_logical_devices('GPU')
        if force_use_gpu and not gpu_devices:
            raise Exception("GPU devices not found")
        c_log.info(device_list_summary(gpu_devices))
        try:
            atexit.register(strategy._extended._cross_device_ops._pool.close)  # type: ignore
            atexit.register(strategy._extended._host_cross_device_ops._pool.close)  # type: ignore
        except AttributeError:
            pass
    return strategy


def get_tpu_strategy_inner(tpu_name):
    from cloud_tpu_client import Client
    c_log.debug("get_tpu_strategy:: init TPUClusterResolver")
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
    c_log.debug("get_tpu_strategy:: experimental_connect_to_cluster")
    tf.config.experimental_connect_to_cluster(resolver)
    c_log.debug("get_tpu_strategy:: initialize_tpu_system")
    tf.tpu.experimental.initialize_tpu_system(resolver)
    c_log.debug("get_tpu_strategy:: init TPUStrategy")
    strategy = tf.distribute.TPUStrategy(resolver)
    c_log.debug("get_tpu_strategy:: init Client")
    c = Client(tpu=tpu_name)
    # c.configure_tpu_version(tf.__version__, restart_type='ifNeeded')
    return strategy
