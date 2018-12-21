from utils import *
import sys

LEARNING_RATE = 0.01

def mutiliclass(train, dev):
    w_len = train.shape[1]
    w = np.zeros(w_len)
    for sample, label in train:
        pred = sample.dot(w)


    return w


if __name__ == '__main__':

    fname = sys.argv[1] if len(sys.argv) > 1 else 'data/letters.train.data'
    l2i, i2l, vectors_labels = read_vectors_file(fname)

    train_idx = int(vectors_labels.shape[1] * 0.8)
    train = vectors_labels[:train_idx]
    dev = vectors_labels[train_idx:]

    #w = mutiliclass(train, dev)
