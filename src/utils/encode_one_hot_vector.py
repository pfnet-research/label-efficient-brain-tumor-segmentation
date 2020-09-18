from chainer.backends import cuda


def _encode_one_hot_vector_core(x, nb_class):
    xp = cuda.get_array_module(x)
    batch, h, w, d = x.shape

    res = xp.zeros((batch, nb_class, h, w, d), dtype=xp.float32)
    x = x.reshape(batch, -1)
    for i in range(batch):
        y = xp.identity(nb_class, dtype=xp.float32)[x[i]]
        res[i] = xp.swapaxes(y, 0, 1).reshape((nb_class, h, w, d))
    return res


def encode_one_hot_vector(x, nb_class):
    if isinstance(x, cuda.ndarray):
        with x.device:
            return _encode_one_hot_vector_core(x, nb_class)
    else:
        return _encode_one_hot_vector_core(x, nb_class)
