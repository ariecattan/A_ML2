import numpy as np



def read_vectors_file(fname):
    out = np.loadtxt(fname, dtype=np.str)
    out = out[:, 1:]

    letters = list(set([x[0] for x in out]))
    l2i = {l:i for i, l in enumerate(letters)}
    i2l = {i:l for l, i in l2i.items()}

    labels = np.array(map(lambda x: l2i[x], out[:, 0]))
    vectors = out[:, -8*16:],

    return l2i, i2l, zip(vectors, labels)