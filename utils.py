import numpy as np
X_INPUT = 128
CATEGORIES = 26

def dic_files(labels):
    l2i = {l:i for i, l in enumerate(list(set(labels)))}
    i2l = {i: l for l, i in l2i.items()}
    return l2i, i2l

def read_vectors_file(fname):
    out = np.loadtxt(fname, dtype=np.str)
    out = out[:, 1:]

    labels = out[:, 0]
    vectors = out[:, -8*16:]
    vectors = [list(map(float, vec)) for vec in vectors]
    vectors = np.array(vectors)

    return vectors, labels


def accuracy(data, label, params):
    preds = data.dot(params)
    preds = np.argmax(preds, axis=1)
    return np.mean(preds == label)

def sample_to_feature(x, y):
    shape = X_INPUT * CATEGORIES
    feat = np.zeros(shape)
    feat[y * X_INPUT:(y+1) * X_INPUT] = x
    return feat


def accuracy_struc(data, label, params):
    counter = 0.0
    for sample, target in zip(data, label):
        feats = np.array([sample_to_feature(sample, y) for y in range(CATEGORIES)])
        pred = np.argmax(feats.dot(params))
        if pred == target:
            counter += 1

    return counter / len(data)