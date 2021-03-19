import msgpack
import msgpack_numpy

msgpack_numpy.patch()

MAX_MSGPACK_LEN = 1000000000


class MsgPackSerializer(object):

    @staticmethod
    def dumps(obj):
        return msgpack.dumps(obj, use_bin_type=True)

    @staticmethod
    def loads(buf):
        return msgpack.loads(
            buf,
            raw=False,
            max_bin_len=MAX_MSGPACK_LEN,
            max_array_len=MAX_MSGPACK_LEN,
            max_map_len=MAX_MSGPACK_LEN,
            max_str_len=MAX_MSGPACK_LEN
        )
