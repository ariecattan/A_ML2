from utils import *
import sys
import operator
from collections import defaultdict, Counter
import time
from sklearn.utils import shuffle

start = time.time()

LEARNING_RATE = 1
EPOCHS = 10
LETTERS = 27

print("Structured Multiclass Perceptron With Previous Label")

mode = sys.argv[1] if len(sys.argv) > 1 else '-t' # '-t'
arg1 = sys.argv[2] if len(sys.argv) > 2 else 'data/letters.train.data'
arg2 = sys.argv[3] if len(sys.argv) > 3 else 'params.txt'
arg3 = sys.argv[4] if len(sys.argv) > 4 else None

#train -t train_data params_file
#predict -p  dev_data params output_file

TRAIN = True if mode == '-t' else False


data = files_to_word(arg1)
params_file = arg2



letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '$']
letters_to_tag = letters[:-1]
l2i = {l:i for i, l in enumerate(letters)}
i2l = {i:l for l, i in l2i.items()}

bigrams = [l1 + l2 for l1 in letters for l2 in letters]
bigram2i = {bi:i for i, bi in enumerate(bigrams)}
i2bigram = {i:bi for bi, i in bigram2i.items()}



def feature_extract(x, y, prev_y):
    shape = X_INPUT * CATEGORIES + LETTERS ** 2
    feat = np.zeros(shape)
    feat[y * X_INPUT:(y + 1) * X_INPUT] = x

    big_id = bigram2i[prev_y + i2l[y]]
    feat[X_INPUT * CATEGORIES + big_id] = 1

    return feat


def multiclass(train_data):
    params = np.zeros(X_INPUT * CATEGORIES + LETTERS ** 2)
    for i in range(EPOCHS):
        print("Epoch " + str(i))
        train_data = shuffle(train_data)
        for num_word, word in enumerate(train_data):
            prev_y = '$'
            for letter in word:
                vec = letter[-8*16:]
                label = letter[0]
                label_id = l2i[label]
                features = np.array([feature_extract(vec, lab, prev_y) for lab in range(LETTERS)])
                pred = np.argmax(features.dot(params))
                if pred != label_id:
                    params = params + features[label_id] - features[pred]

                prev_y = label

    return params


def get_score_one_letter(vec, prev_letter, params):
    features = np.array([feature_extract(vec, lab, prev_letter) for lab in range(LETTERS)])
    pred = features.dot(params)

    output = {}
    for i in range(len(pred)):
        letter = i2l[i]
        output[letter] = pred[i]

    return output


def predict(data, params):
    output = []
    for word in data:
        prev_char = '$'

        Viterbi = []
        Labels = []

        first_letter = word[0][-8*16:]
        sc = get_score_one_letter(first_letter, prev_char, params)
        Viterbi.append(sc)
        labs = {l:prev_char for l in letters}
        Labels.append(labs)

        for i, letter in enumerate(word[1:]):
            letter_vec = letter[-8*16:]
            results = [get_score_one_letter(letter_vec, prev, params) for prev in letters]

            scores = {}
            prev_tags = {}
            for tag in letters:
                values = []
                for prev_char in letters:
                    prev_char_id = l2i[prev_char]
                    sc = results[prev_char_id][tag]
                    score = Viterbi[i - 1][prev_char] + sc
                    values.append([score, prev_char])

                max_score, best_prev = max(values)
                scores[tag] = max_score
                prev_tags[tag] = best_prev

            Viterbi.append(scores)
            Labels.append(prev_tags)


        last_tag = max(Viterbi[-1].items(), key=operator.itemgetter(1))[0]

        preds = []
        preds.append(last_tag)


        for j in range(len(word)-2, 0, -1):
            back_tag = Labels[j][last_tag]
            last_tag = back_tag
            preds.append(last_tag)

        preds.reverse()
        output.append(preds)

    return output




if __name__ == '__main__':
    a = 0

    if TRAIN:
        w = multiclass(data)
        np.savetxt(params_file, w, fmt='%d')

    else:
        w = np.loadtxt(params_file)
        words = data[:5]
        #preds = predict(words, w)


    end = time.time() - start

    print(end)


