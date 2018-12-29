from utils import *
import sys
from sklearn.utils import shuffle

LEARNING_RATE = 1
EPOCHS = 3


def multiclass(train_samples, train_labels):
    params = np.zeros(X_INPUT * CATEGORIES)
    for i in range(EPOCHS):
        X, Y = shuffle(train_samples, train_labels)
        print(i)
        for sample, label in zip(X, Y):
            features = np.array([sample_to_feature(sample, lab) for lab in range(CATEGORIES)])
            pred = np.argmax(features.dot(params))
            if pred != label:
                params = params + features[label] - features[pred]

        accD = accuracy_struc(dev_samples, dev_labels, params)
        print('Accuracy on the dev set: {}'.format(accD))
    return params




if __name__ == '__main__':
    print("Structured Multiclass Perceptron")

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

    w = multiclass(train_samples, train_labels)


