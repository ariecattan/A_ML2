from utils import *
import sys
from sklearn.utils import shuffle

LEARNING_RATE = 1
EPOCHS = 50

def multiclass(train_samples, train_labels, shape):
    params = np.random.rand(shape[0], shape[1])
    for i in range(EPOCHS):
        X, Y = shuffle(train_samples, train_labels)
        print(i)
        for sample, label in zip(X, Y):
            pred = np.argmax(sample.dot(params))
            if pred != label:
                params[:, label] += sample
                params[:, pred] -=  sample

        accD = accuracy(dev_samples, dev_labels, params)
        print('Accuracy on the dev set: {}'.format(accD))
    return params



if __name__ == '__main__':
    print("Multiclass Perceptron")

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
    shape = (input_vector, categories)


    w = multiclass(train_samples, train_labels, shape)


