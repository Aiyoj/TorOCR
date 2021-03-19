import os
import six
import threading
import numpy as np

from abc import ABC
from datetime import datetime
from collections import deque


class DataFlowReentrantGuard(object):
    def __init__(self):
        self._lock = threading.Lock()

    def __enter__(self):
        self._succ = self._lock.acquire(False)
        if not self._succ:
            raise threading.ThreadError("This DataFlow is not reentrant!")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()
        return False


class DataFlow(ABC):
    def __init__(self):
        self._start_done = False

    def __iter__(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def start(self):
        assert not self._start_done, "start() was called twice!"
        self._start_done = True


class RepeatedData(DataFlow):
    def __init__(self, ds: DataFlow, n_epoch: int = None):
        super(RepeatedData, self).__init__()

        assert (n_epoch > 0 if isinstance(n_epoch, int) else n_epoch is None), n_epoch

        self.n_epoch = n_epoch
        self.ds = ds

    def __len__(self):
        if self.n_epoch is None:
            raise NotImplementedError("__len__() is unavailable for infinite dataflow")

        return len(self.ds) * self.n_epoch

    def __iter__(self):
        if self.n_epoch is None:
            while True:
                yield from self.ds
        else:
            for _ in range(self.n_epoch):
                yield from self.ds

    def start(self):
        self.ds.start()


class RepeatedDataPoint(DataFlow):
    def __init__(self, ds: DataFlow, n_repeat: int):
        super(RepeatedDataPoint, self).__init__()

        assert n_repeat > 0, n_repeat

        self.n_repeat = n_repeat
        self.ds = ds

    def __len__(self):
        return len(self.ds) * self.n_repeat

    def __iter__(self):
        for dp in self.ds:
            for _ in range(self.n_repeat):
                yield dp

    def start(self):
        self.ds.start()


class BatchData(DataFlow):
    def __init__(self, ds: DataFlow, batch_size: int, remainder: bool = False, use_list: bool = False):
        super(BatchData, self).__init__()

        assert batch_size > 0, batch_size

        self.ds = ds
        if not remainder:
            try:
                assert batch_size <= len(ds)
            except NotImplementedError:
                pass
        self.batch_size = batch_size
        self.remainder = remainder
        self.use_list = use_list

    def __len__(self):
        ds_size = len(self.ds)
        div = ds_size // self.batch_size
        rem = ds_size % self.batch_size
        if rem == 0:
            return div
        return div + int(self.remainder)

    def __iter__(self):
        holder = []
        for data in self.ds:
            holder += [data]
            if len(holder) == self.batch_size:
                yield self.aggregate_batch(holder, self.use_list)
                del holder[:]
        if self.remainder and len(holder) > 0:
            yield self.aggregate_batch(holder, self.use_list)

    @staticmethod
    def batch_numpy(data_list):
        data = data_list[0]
        if isinstance(data, six.integer_types):
            dtype = "int32"
        elif type(data) == bool:
            dtype = "bool"
        elif type(data) == float:
            dtype = "float32"
        elif isinstance(data, (six.binary_type, six.text_type)):
            dtype = "str"
        else:
            try:
                dtype = data.dtype
            except AttributeError:
                raise TypeError("Unsupported type to batch: {}".format(type(data)))

        try:
            return np.asarray(data_list, dtype=dtype)
        except Exception as e:  # noqa
            print("Cannot batch data. Perhaps they are of inconsistent shape?")

    @staticmethod
    def aggregate_batch(data_holder, use_list=False):
        first_dp = data_holder[0]
        if isinstance(first_dp, (list, tuple)):
            result = []
            for k in range(len(first_dp)):
                data_list = [x[k] for x in data_holder]
                if use_list:
                    result.append(data_list)
                else:
                    result.append(BatchData.batch_numpy(data_list))
        elif isinstance(first_dp, dict):
            result = {}
            for key in first_dp.keys():
                data_list = [x[key] for x in data_holder]
                if use_list:
                    result[key] = data_list
                else:
                    result[key] = BatchData.batch_numpy(data_list)
        else:
            raise ValueError("Data point has to be list/tuple/dict. Got {}".format(type(first_dp)))
        return result

    def start(self):
        self.ds.start()


class ConcatData(DataFlow):
    def __init__(self, dsl):
        super(ConcatData, self).__init__()
        self.dsl = dsl

    def start(self):
        for ds in self.dsl:
            ds.start()

    def __len__(self):
        return sum(len(x) for x in self.dsl)

    def __iter__(self):
        for ds in self.dsl:
            yield from ds


class CacheData(DataFlow):
    def __init__(self, ds: DataFlow, shuffle: bool = False, seed: int = None):
        super(CacheData, self).__init__()
        self.ds = ds
        self.shuffle = shuffle
        self._lock = threading.Lock()
        if self.shuffle:
            if seed is not None:
                self.rng = np.random.RandomState(seed)
            else:
                seed = (id(self) + os.getpid() + int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
                self.rng = np.random.RandomState(seed)
        self.buffer = []

    def __len__(self):
        return self.ds.__len__()

    def __iter__(self):
        with self._lock:
            if len(self.buffer):
                if self.shuffle:
                    self.rng.shuffle(self.buffer)
                yield from self.buffer
            else:
                for dp in self.ds:
                    yield dp
                    self.buffer.append(dp)

    def start(self):
        self.ds.start()


class LocallyShuffleData(DataFlow):

    def __init__(self, ds, buffer_size, shuffle_interval=None):
        super(LocallyShuffleData, self).__init__()
        self.ds = ds
        self.q = deque(maxlen=buffer_size)
        if shuffle_interval is None:
            shuffle_interval = int(buffer_size // 3)
        self.shuffle_interval = shuffle_interval
        self.ds = ds

    def start(self):
        self._guard = DataFlowReentrantGuard()
        self.ds.start()
        seed = (id(self) + os.getpid() + int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
        self.rng = np.random.RandomState(seed)
        self._iter_cnt = 0
        # self._inf_iter = iter(self.ds)

    def __len__(self):
        return len(self.ds)

    def __iter__(self):
        with self._guard:
            for dp in self.ds:
                self._iter_cnt = (self._iter_cnt + 1) % self.shuffle_interval
                # fill queue
                if self._iter_cnt == 0:
                    self.rng.shuffle(self.q)

                if self.q.maxlen == len(self.q):
                    yield self.q.popleft()
                self.q.append(dp)
            while len(self.q) != 0:
                yield self.q.popleft()
