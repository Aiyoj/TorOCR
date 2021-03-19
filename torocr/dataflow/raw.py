import os
import numpy as np

from datetime import datetime

from torocr.dataflow.common import DataFlow


class DataFromQueue(DataFlow):
    def __init__(self, queue):
        super(DataFromQueue, self).__init__()
        self.queue = queue

    def __iter__(self):
        while True:
            yield self.queue.get()

    def __len__(self):
        raise NotImplementedError

    def start(self):
        raise NotImplementedError


class DataFromList(DataFlow):
    def __init__(self, inputs, shuffle=True, seed=None):
        super(DataFromList, self).__init__()
        self.inputs = inputs
        self.shuffle = shuffle
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            seed = (id(self) + os.getpid() + int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
            self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.inputs)

    def __iter__(self):
        if not self.shuffle:
            yield from self.inputs
        else:
            indices = np.arange(len(self.inputs))
            self.rng.shuffle(indices)
            for i in indices:
                yield self.inputs[i]


class DataFromGenerator(DataFlow):
    def __init__(self, gen):
        super(DataFromGenerator, self).__init__()
        self._gen = gen

    def __iter__(self):
        if not callable(self._gen):
            yield from self._gen
        else:
            yield from self._gen()

    def __len__(self):
        return len(self._gen)


class DataFromIterable(DataFlow):
    def __init__(self, iterable):
        super(DataFromIterable, self).__init__()
        self._itr = iterable
        try:
            self._len = len(iterable)
        except Exception as e:
            self._len = None

    def __len__(self):
        if self._len is None:
            raise NotImplementedError
        return self._len

    def __iter__(self):
        yield from self._itr
