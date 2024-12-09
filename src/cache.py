import json
import os
from pickle import UnpicklingError
from typing import TypeVar, Generic

from cpath import cache_path, data_path, json_cache_path


def load_from_pickle(name):
    pickle_name = "{}.pickle".format(name)
    path = os.path.join(cache_path, pickle_name)
    try:
        obj = pickle.load(open(path, "rb"))
    except UnpicklingError:
        print("pickle name:", name)
        raise

    return obj


def save_to_pickle(obj, name: str):
    if not type(name) == str:
        raise TypeError("You passed {} as name and {} as obj".format(name, obj))
    pickle_name = "{}.pickle".format(name)
    path = os.path.join(cache_path, pickle_name)
    pickle.dump(obj, open(path, "wb"))


def load_pickle_from(path):
    return pickle.load(open(path, "rb"))


def load_cache(name):
    pickle_name = "{}.pickle".format(name)
    path = os.path.join(cache_path, pickle_name)
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    else:
        return None


def function_cache_wrap(fn, name):
    pickle_name = "{}.pickle".format(name)
    path = os.path.join(cache_path, pickle_name)
    def wrap_fn():
        if os.path.exists(path):
            print("Using cached value {}".format(name))
            ret = pickle.load(open(path, "rb"))
        else:
            ret = fn()

        save_to_pickle(ret, name)
        return ret
    return wrap_fn


def dump_to_json(obj, name):
    pickle_name = "{}.json".format(name)
    path = os.path.join(json_cache_path, pickle_name)
    return json.dump(obj, open(path, "w"))


def load_json_cache(name):
    pickle_name = "{}.json".format(name)
    path = os.path.join(json_cache_path, pickle_name)
    if os.path.exists(path):
        return json.load(open(path, "r"))
    else:
        return None



class StreamPickler:
    def __init__(self, name, save_per):
        self.idx = 0
        self.current_chunk = []
        self.save_per = save_per
        self.save_prefix = os.path.join(data_path, "stream_pickled", name)

    def flush(self):
        if len(self.current_chunk) == 0:
            return
        save_name = self.save_prefix + str(self.idx)
        pickle.dump(self.current_chunk, open(save_name, "wb"))
        self.current_chunk = []
        self.idx += 1

    def add(self, inst):
        self.current_chunk.append(inst)
        if len(self.current_chunk) == self.save_per:
            self.flush()


class StreamPickler2(StreamPickler):
    def __init__(self, name, save_per):
        super(StreamPickler2, self).__init__(name, save_per)
        self.save_prefix = os.path.join(data_path, "stream_pickled", name, "data_")


class StreamPickleReader:
    def __init__(self, name, pickle_idx = 0):
        self.pickle_idx = pickle_idx
        self.current_chunk = []
        self.chunk_idx = 0
        self.save_prefix = os.path.join(data_path, "stream_pickled", name)
        self.acc_item = 0

    def get_item(self):
        if self.chunk_idx >= len(self.current_chunk):
            self.get_new_chunk()

        item = self.current_chunk[self.chunk_idx]
        self.chunk_idx += 1
        self.acc_item += 1
        return item

    def limited_has_next(self, limit):
        if self.acc_item < limit:
            return self.has_next()
        else:
            return False

    def get_new_chunk(self):
        save_name = self.next_chunk_path()
        self.current_chunk = pickle.load(open(save_name, "rb"))
        assert len(self.current_chunk) > 0
        self.chunk_idx = 0
        self.pickle_idx += 1

    def next_chunk_path(self):
        return self.save_prefix + str(self.pickle_idx)

    def has_next(self):
        if self.chunk_idx +1 < len(self.current_chunk):
            return True

        return os.path.exists(self.next_chunk_path())


import pickle


class DumpPickle:
    def __init__(self, out_path):
        self.out_f = open(out_path, "wb")
        self.dict_out_path = out_path + ".dict"
        self.loc_dict = {}

    def dump(self, name, obj):
        fp_loc = self.out_f.tell()
        pickle.dump(obj, self.out_f)
        fp_ed = self.out_f.tell()
        self.loc_dict[name] = (fp_loc, fp_ed)

    def close(self):
        self.out_f.close()
        pickle.dump(self.loc_dict, open(self.dict_out_path, "wb"))


class DumpPickleLoader:
    def __init__(self, out_path):
        self.out_f = open(out_path, "rb")
        dict_out_path = out_path + ".dict"
        self.loc_dict = pickle.load(open(dict_out_path, "rb"))

    def load(self, name):
        st, ed = self.loc_dict[name]
        self.out_f.seek(st)
        data = self.out_f.read(ed - st)
        return pickle.loads(data)


def save_list_to_jsonl(item_list, save_path):
    f = open(save_path, "w")
    for item in item_list:
        s = json.dumps(item)
        f.write(s + "\n")
    f.close()


def save_list_to_jsonl_w_fn(item_list, save_path, to_json):
    f = open(save_path, "w")
    for item in item_list:
        s = json.dumps(to_json(item))
        f.write(s + "\n")
    f.close()


def load_list_from_jsonl(save_path, from_json):
    f = open(save_path, "r")
    return [from_json(json.loads(line)) for line in f]


def named_tuple_to_json(obj):
    _json = {}
    if isinstance(obj, tuple):
        datas = obj._asdict()
        for data in datas:
            _json[data] = (datas[data])
    return _json


T = TypeVar('T')


class StrItem(Generic[T]):
    def __init__(self, s, item):
        self.s = s
        self.item = item

    def to_json(self):
        return {
            'string': self.s,
            'item': self.item.to_json(),
        }

    @classmethod
    def from_json(cls, j, from_json_for_item):
        return StrItem(j['string'], from_json_for_item(j['item']))

    def to_tuple(self):
        return self.s, self.item