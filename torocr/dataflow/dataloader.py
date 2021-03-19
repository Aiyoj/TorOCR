import os
import re
import zmq
import time
import random
import atexit
import weakref
import threading
import numpy as np
import collections
import multiprocessing as mp
import zmq.decorators as zmqd

from torocr.dataflow.sampler import RandomSampler, SequentialSampler, BatchSampler
from torocr.utils.common_utils import del_weakref
from torocr.utils.zmq_utils import auto_bind, multi_socket
from torocr.dataflow.serialize import MsgPackSerializer

numpy_type_map = {
    "float64": np.float64,
    "float32": np.float32,
    "float16": np.float16,
    "int64": np.int64,
    "int32": np.int32,
    "int16": np.int16,
    "int8": np.int8,
    "uint8": np.uint8,
}


def default_collate(batch):
    """
    Puts each data field into a ndarray with outer dimension batch size
    """

    error_msg = "batch must contain ndarrays, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if elem_type.__module__ == "numpy" and elem_type.__name__ != "str_" and elem_type.__name__ != "string_":
        elem = batch[0]
        if elem_type.__name__ == "ndarray":
            # array of string classes and object
            if re.search("[SaUO]", elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))
            try:
                return np.stack([np.array(b, elem.dtype) for b in batch], 0)
            except Exception as e:
                return batch
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith("float") else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int):
        return np.int64(batch)
    elif isinstance(batch[0], float):
        return np.float64(batch)
    elif isinstance(batch[0], str):
        return batch
    elif isinstance(batch[0], list):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0].keys()}

    raise TypeError((error_msg.format(type(batch[0]))))


class DataLoader(object):
    class _Dispatcher(threading.Thread):
        def __init__(self, dataset, batch_sampler, batch_size, num_workers, sink_address, hwm, seed=None):
            super(DataLoader._Dispatcher, self).__init__()

            self.dataset = dataset
            self.batch_sampler = batch_sampler
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.sink_address = sink_address
            self.hwm = hwm

            self.task_pending_lock = threading.Condition()
            self.sem = mp.Semaphore(0)
            self.num_concurrent_socket = self.num_workers
            self.daemon = True
            self.not_init = True

            self.processes = []

            self.pipe = []

            if seed is not None:
                np.random.seed(seed)

        @zmqd.context()
        @multi_socket(zmq.PUSH, num_socket="num_concurrent_socket")
        def run(self, ctx, backend_socks):
            rand_backend_socket = None
            addr_backend_list = []
            for i, backend_sock in enumerate(backend_socks):
                backend_sock.set_hwm(self.hwm)
                addr, pipe_dir = auto_bind(backend_sock, "dataflow-map")
                addr_backend_list.append(addr)
                self.pipe.append(pipe_dir)

            for i in range(self.num_workers):
                process = DataLoader._Worker(
                    dataset=self.dataset, worker_id=i, worker_addresses=addr_backend_list,
                    sink_address=self.sink_address, hwm=self.hwm
                )
                self.processes.append(process)
                process.start()

            self.sampler_iter = iter(self.batch_sampler)
            self.not_init = True
            flag = True
            while True:
                batch_indices = next(self.sampler_iter, None)
                if batch_indices is None:
                    time.sleep(0.001)
                    self.sampler_iter = iter(self.batch_sampler)
                    flag = True
                    self.sem.acquire()
                    continue
                else:
                    if self.not_init:
                        self.task_pending_lock.acquire()
                        self.task_pending_lock.notify()
                        self.not_init = False
                        self.task_pending_lock.release()

                    if flag:
                        flag = False
                        self.sem.acquire()

                if len(backend_socks) == 1:
                    rand_backend_socket = backend_socks[0]
                else:
                    rand_backend_socket = np.random.choice([b for b in backend_socks if b != rand_backend_socket])
                rand_backend_socket.send(MsgPackSerializer.dumps(batch_indices), copy=False)
                time.sleep(0.001)

    class _Worker(mp.Process):
        def __init__(self, dataset, worker_id, worker_addresses, sink_address, hwm):
            super(DataLoader._Worker, self).__init__()

            self.dataset = dataset
            self.worker_id = worker_id
            self.worker_addresses = worker_addresses
            self.sink_address = sink_address
            self.hwm = hwm

            self.num_workers = len(self.worker_addresses)

        @zmqd.context()
        @zmqd.socket(zmq.PUSH)
        @zmqd.socket(zmq.PULL)
        def run(self, ctx, sink, receiver):
            receiver.set_hwm(self.hwm)
            receiver.connect(self.worker_addresses[self.worker_id])

            sink.set_hwm(self.hwm * self.num_workers)
            sink.connect(self.sink_address)

            while True:
                batch_indices = MsgPackSerializer.loads(receiver.recv(copy=False))

                batch = []
                for idx in batch_indices:
                    batch.append(self.dataset[idx])
                samples = default_collate(batch)
                sink.send(MsgPackSerializer.dumps(samples), copy=False)

    def __init__(self, dataset, batch_size=1, buffer_size=1, shuffle=True, sampler=None,
                 batch_sampler=None, num_workers=1, collate_fn=None, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.pipe = []

        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError("batch_sampler is mutually exclusive with batch_size, shuffle, sampler, and drop_last")

        # if sampler is not None and shuffle:
        #     raise ValueError("sampler is mutually exclusive with shuffle")

        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative; use num_workers=0 to disable multiprocessing.")

        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.sampler = sampler
        self.batch_sampler = batch_sampler

        assert self.num_workers > 0

        atexit.register(del_weakref, weakref.ref(self))

        self.context = zmq.Context()
        self.sink = self.context.socket(zmq.PULL)
        self.sink.set_hwm(self.buffer_size * self.num_workers)
        addr, pipe_dir = auto_bind(self.sink, "dataflow-sink")
        self.sink_addr = addr

        self.pipe.append(pipe_dir)

        self.dispatcher = DataLoader._Dispatcher(
            dataset=self.dataset, batch_sampler=self.batch_sampler,
            num_workers=self.num_workers, batch_size=self.batch_size,
            sink_address=self.sink_addr, hwm=self.buffer_size
        )
        self.dispatcher.start()

    def __iter__(self):
        if self.dispatcher.not_init:
            self.dispatcher.task_pending_lock.acquire()
            self.dispatcher.task_pending_lock.wait()
            self.dispatcher.task_pending_lock.release()

        self.dispatcher.sem.release()
        for _ in range(len(self)):
            batch = MsgPackSerializer.loads(self.sink.recv(copy=False))
            if self.collate_fn is not None:
                batch = self.collate_fn(batch)

            yield batch
        self.dispatcher.sem.release()

    def __len__(self):
        return len(self.batch_sampler)

    def __del__(self):
        try:
            if not self.context.closed:
                self.sink.close(0)
                self.context.destroy(0)

            for x in self.dispatcher.processes:
                x.terminate()
                x.join(5)

            for p in self.pipe:
                if os.path.exists(p):
                    os.remove(p)

            for p in self.dispatcher.pipe:
                if os.path.exists(p):
                    os.remove(p)
            print("{} successfully cleaned-up.".format(type(self).__name__))
        except Exception as e:
            pass


class MSDataLoader(object):
    class _Dispatcher(threading.Thread):
        def __init__(self, dataset, batch_sampler, batch_size, num_workers, sink_address, hwm, ms, seed=None):
            super(MSDataLoader._Dispatcher, self).__init__()

            self.dataset = dataset
            self.batch_sampler = batch_sampler
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.sink_address = sink_address
            self.hwm = hwm
            self.ms = ms

            self.task_pending_lock = threading.Condition()
            self.sem = mp.Semaphore(0)
            self.num_concurrent_socket = self.num_workers
            self.daemon = True
            self.not_init = True

            self.processes = []
            self.pipe = []

            if seed is not None:
                np.random.seed(seed)

        @zmqd.context()
        @multi_socket(zmq.PUSH, num_socket="num_concurrent_socket")
        def run(self, ctx, backend_socks):
            rand_backend_socket = None
            addr_backend_list = []
            for i, backend_sock in enumerate(backend_socks):
                backend_sock.set_hwm(self.hwm)
                addr, pipe_dir = auto_bind(backend_sock, "dataflow-map")
                addr_backend_list.append(addr)
                self.pipe.append(pipe_dir)

            for i in range(self.num_workers):
                process = MSDataLoader._Worker(
                    dataset=self.dataset, worker_id=i, worker_addresses=addr_backend_list,
                    sink_address=self.sink_address, hwm=self.hwm, ms=self.ms
                )
                self.processes.append(process)
                process.start()

            self.sampler_iter = iter(self.batch_sampler)
            self.not_init = True
            flag = True
            while True:
                batch_indices = next(self.sampler_iter, None)
                if batch_indices is None:
                    time.sleep(0.001)
                    self.sampler_iter = iter(self.batch_sampler)
                    flag = True
                    self.sem.acquire()
                    continue
                else:
                    if self.not_init:
                        self.task_pending_lock.acquire()
                        self.task_pending_lock.notify()
                        self.not_init = False
                        self.task_pending_lock.release()

                    if flag:
                        flag = False
                        self.sem.acquire()

                if len(backend_socks) == 1:
                    rand_backend_socket = backend_socks[0]
                else:
                    rand_backend_socket = np.random.choice([b for b in backend_socks if b != rand_backend_socket])
                rand_backend_socket.send(MsgPackSerializer.dumps(batch_indices), copy=False)
                time.sleep(0.001)

    class _Worker(mp.Process):
        def __init__(self, dataset, worker_id, worker_addresses, sink_address, hwm, ms):
            super(MSDataLoader._Worker, self).__init__()

            self.dataset = dataset
            self.worker_id = worker_id
            self.worker_addresses = worker_addresses
            self.sink_address = sink_address
            self.hwm = hwm
            self.ms = ms

            self.num_workers = len(self.worker_addresses)

        @zmqd.context()
        @zmqd.socket(zmq.PUSH)
        @zmqd.socket(zmq.PULL)
        def run(self, ctx, sink, receiver):
            receiver.set_hwm(self.hwm)
            receiver.connect(self.worker_addresses[self.worker_id])

            sink.set_hwm(self.hwm * self.num_workers)
            sink.connect(self.sink_address)

            while True:
                batch_indices = MsgPackSerializer.loads(receiver.recv(copy=False))
                size = random.choice(self.ms)
                self.dataset.dynamic_size = size

                batch = []
                for idx in batch_indices:
                    batch.append(self.dataset[idx])
                samples = default_collate(batch)
                sink.send(MsgPackSerializer.dumps(samples), copy=False)

    def __init__(self, dataset, ms=[640, 672, 704, 736, 768, 800, 832, 864, 896], batch_size=1, buffer_size=4,
                 shuffle=True, sampler=None,
                 batch_sampler=None, num_workers=1, collate_fn=None, drop_last=False):
        self.dataset = dataset
        self.ms = ms
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.pipe = []

        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError("batch_sampler is mutually exclusive with batch_size, shuffle, sampler, and drop_last")

        # if sampler is not None and shuffle:
        #     raise ValueError("sampler is mutually exclusive with shuffle")

        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative; use num_workers=0 to disable multiprocessing.")

        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.sampler = sampler
        self.batch_sampler = batch_sampler

        assert self.num_workers > 0

        atexit.register(del_weakref, weakref.ref(self))

        self.context = zmq.Context()
        self.sink = self.context.socket(zmq.PULL)
        self.sink.set_hwm(self.buffer_size * self.num_workers)
        addr, pipe_dir = auto_bind(self.sink, "dataflow-sink")
        self.sink_addr = addr
        self.pipe.append(pipe_dir)

        self.dispatcher = MSDataLoader._Dispatcher(
            dataset=self.dataset, batch_sampler=self.batch_sampler,
            num_workers=self.num_workers, batch_size=self.batch_size,
            sink_address=self.sink_addr, hwm=self.buffer_size, ms=self.ms
        )
        self.dispatcher.start()

    def __iter__(self):
        if self.dispatcher.not_init:
            self.dispatcher.task_pending_lock.acquire()
            self.dispatcher.task_pending_lock.wait()
            self.dispatcher.task_pending_lock.release()

        self.dispatcher.sem.release()
        for _ in range(len(self)):
            batch = MsgPackSerializer.loads(self.sink.recv(copy=False))
            if self.collate_fn is not None:
                batch = self.collate_fn(batch)

            yield batch
        self.dispatcher.sem.release()

    def __len__(self):
        return len(self.batch_sampler)

    def __del__(self):
        try:
            if not self.context.closed:
                self.sink.close(0)
                self.context.destroy(0)

            for x in self.dispatcher.processes:
                x.terminate()
                x.join(5)

            for p in self.pipe:
                if os.path.exists(p):
                    os.remove(p)

            for p in self.dispatcher.pipe:
                if os.path.exists(p):
                    os.remove(p)
            print("{} successfully cleaned-up.".format(type(self).__name__))
        except Exception as e:
            pass
