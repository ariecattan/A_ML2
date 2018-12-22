from utils import *
import sys
from sklearn.utils import shuffle

LEARNING_RATE = 1
EPOCHS = 10

def multiclass(train_samples, train_labels, shape):
    params = np.random.rand(shape)
    for i in range(EPOCHS):
        X, Y = shuffle(train_samples, train_labels)
        print(i)
        for sample, label in zip(X, Y):
            pred = np.argmax(sample.dot(params))
            params += sample_to_feature(sample, label) - sample_to_feature(sample, pred)
    return params

def sample_to_feature(x, y):
    shape = len(x) * 26
    feat = np.zeros(shape)


if __name__ == '__main__':

    tname = sys.argv[1] if len(sys.argv) > 1 else 'data/letters.train.data'
    dname = sys.argv[2] if len(sys.argv) > 2 else 'data/letters.test.data'
    train_samples, train_labels = read_vectors_file(tname)
    dev_samples, dev_labels = read_vectors_file(dname)
    l2i, i2l = dic_files(train_labels)

    train_labels = list(map(lambda x: l2i[x], train_labels))
    dev_labels = list(map(lambda x: l2i[x], dev_labels))

    train = train_samples, train_labels

    input_vector = train_samples.shape[1]
    categories = len(l2i)
    shape = input_vector * categories

    #w = np.random.rand(input_vector, categories)

    w = multiclass(train_samples, train_labels, shape)
    #for i in range(EPOCHS):
        #w = multiclass(train)

    accD = accuracy(dev_samples, dev_labels, w)
    accT = accuracy(train_samples, train_labels, w)