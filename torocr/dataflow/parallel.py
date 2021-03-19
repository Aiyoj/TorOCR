import zmq
import random
import atexit
import weakref
import threading
import multiprocessing as mp
import zmq.decorators as zmqd

from torocr.utils.common_utils import del_weakref
from torocr.utils.zmq_utils import auto_bind, multi_socket
from torocr.dataflow.common import DataFlow, BatchData
from torocr.dataflow.serialize import MsgPackSerializer
from torocr.dataflow.common import DataFlowReentrantGuard


class MapAndBatchData(DataFlow):
    class _Dispatcher(threading.Thread):
        def __init__(self, ds, n_worker, map_func, sink_address, hwm, batch_size):
            super(MapAndBatchData._Dispatcher, self).__init__()

            self.ds = ds
            self.n_worker = n_worker
            self.map_func = map_func
            self.sink_address = sink_address
            self.hwm = hwm
            self.batch_size = batch_size

            self.sem = mp.Semaphore(0)
            self.processes = []
            self.num_concurrent_socket = self.n_worker
            self.daemon = True

        @zmqd.context()
        @multi_socket(zmq.PUSH, num_socket="num_concurrent_socket")
        def run(self, ctx, backend_socks):
            rand_backend_socket = None

            for i, backend_sock in enumerate(backend_socks):
                backend_sock.set_hwm(self.hwm)
            addr_backend_list = [auto_bind(b, "dataflow-map") for b in backend_socks]

            self.ds.start()

            for i in range(self.n_worker):
                process = MapAndBatchData._Worker(
                    worker_id=i, worker_addresses=addr_backend_list,
                    sink_address=self.sink_address, map_func=self.map_func,
                    hwm=self.hwm, batch_size=self.batch_size
                )
                self.processes.append(process)
                process.start()

            while True:
                self.sem.acquire()
                for dp in self.ds:
                    rand_backend_socket = random.choice([b for b in backend_socks if b != rand_backend_socket])
                    rand_backend_socket.send(MsgPackSerializer.dumps(dp), copy=False)
                self.sem.acquire()

    class _Worker(mp.Process):
        def __init__(self, worker_id, worker_addresses, sink_address, map_func, hwm, batch_size):
            super(MapAndBatchData._Worker, self).__init__()

            self.worker_id = worker_id
            self.worker_addresses = worker_addresses
            self.sink_address = sink_address
            self.map_func = map_func
            self.hwm = hwm
            self.batch_size = batch_size

            self.daemon = True

        @zmqd.context()
        @zmqd.socket(zmq.PUSH)
        @zmqd.socket(zmq.PULL)
        def run(self, ctx, sink, receiver):
            receiver.set_hwm(self.hwm * self.batch_size)
            receiver.connect(self.worker_addresses[self.worker_id])

            sink.set_hwm(self.hwm)
            sink.connect(self.sink_address)

            batch = []
            while True:
                dp = MsgPackSerializer.loads(receiver.recv(copy=False))

                dp = self.map_func(dp)
                if dp is not None:
                    batch.append(dp)
                    if len(batch) == self.batch_size:
                        dp = BatchData.aggregate_batch(batch, use_list=True)
                        sink.send(MsgPackSerializer.dumps(dp), copy=False)
                        del batch[:]

    def __init__(self, ds, n_worker, map_func, batch_size, buffer_size):
        super(MapAndBatchData, self).__init__()

        if buffer_size is not None:
            assert batch_size < buffer_size

        self.ds = ds
        self.n_worker = n_worker
        self.map_func = map_func
        self.batch_size = batch_size
        if buffer_size is None:
            buffer_size = batch_size * 100
        self.buffer_size = buffer_size
        self.count = 0

        self.is_first = False

    def start(self):
        super(MapAndBatchData, self).start()

        atexit.register(del_weakref, weakref.ref(self))
        self._guard = DataFlowReentrantGuard()

        self.context = zmq.Context()
        self.sink = self.context.socket(zmq.PULL)
        self.sink.set_hwm(self.buffer_size)
        self.sink_addr = auto_bind(self.sink, "dataflow-sink")

        self.dispatcher = MapAndBatchData._Dispatcher(
            ds=self.ds, n_worker=self.n_worker, map_func=self.map_func,
            sink_address=self.sink_addr, hwm=self.buffer_size, batch_size=self.batch_size
        )
        self.dispatcher.start()

    def __del__(self):
        try:
            if not self._start_done:
                return
            if not self.context.closed:
                self.sink.close(0)
                self.context.destroy(0)
            for x in self.dispatcher.processes:
                x.terminate()
                x.join(5)
            print("{} successfully cleaned-up.".format(type(self).__name__))
        except Exception as e:
            pass

    def __iter__(self):
        with self._guard:
            self.dispatcher.sem.release()
            for _ in range(len(self)):
                yield MsgPackSerializer.loads(self.sink.recv(copy=False))
            self.dispatcher.sem.release()

    def __len__(self):
        return len(self.ds)
