from random import shuffle
import re
import numpy as np
import tensorflow as tf

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import save_model


class TextClassifierModel1:

    def __init__(self, file, dictionary=None):
        # load data
        self.x_train, self.y_train, self.x_test, self.y_test = load_training_data(file)
        # load dictionary (or creates it)
        if dictionary is None:
            print("Creating Dictionary...")
            self.dictionary = create_word_dictionary(self.x_train + self.x_test)
            save_dict(self.dictionary)
            print("Dictionary created (size=" + str(len(self.dictionary.keys())) + "), saved as dict.txt.")
        else:
            print("Loading Dictionary")
            self.dictionary = load_word_dictionary(self.dictionary)
            print("Dictionary created")

        # Tokenize data
        self.maxSentenceSize = get_max_sentence_size(self.x_train+self.x_test)
        self.x_train_tokens = translate_to_tokens(self.x_train, self.maxSentenceSize, self.dictionary)
        self.x_test_tokens = translate_to_tokens(self.x_test, self.maxSentenceSize, self.dictionary)

    def initiate_model(self):
        print('Creating model...')
        model = Sequential()
        embedding_size = 3
        model.add(Embedding(input_dim=len(self.dictionary)+1, output_dim=embedding_size, input_length=self.maxSentenceSize, name='layer_embedding'))
        model.add(GRU(units=16, name="gru_1", return_sequences=True))
        model.add(GRU(units=8, name="gru_2", return_sequences=True))
        model.add(GRU(units=4, name="gru_3"))
        model.add(Dense(1, activation='sigmoid', name="dense_1"))
        optimizer = Adam(lr=1e-3)
        model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])
        print('Training model...')
        model.fit(np.array(self.x_train_tokens), np.array(self.y_train), validation_split=0.05, epochs=5, batch_size=32)
        model.summary()
        txt = ["awesome movie", "Terrible movie", "that movie really sucks", "I like that movie"]
        print(translate_to_tokens(txt, self.maxSentenceSize, self.dictionary)[0])
        pred = model.predict(translate_to_tokens(txt), self.maxSentenceSize, self.dictionary)
        print('\n prediction for \n', pred[:, 0])


def load_training_data(file, test_percentage=0.2):
    '''
    Loads data from csv and returns the training/testing sets.
    :param file: comma separated file. First column is text and second one is its classification.
    :param test_percentage: percentage of data for the test set.
    :return: trainingSet,
    '''

    with open(file, 'r', encoding='UTF-8') as f:
        tweets = f.readlines()

    shuffle(tweets) # Mitigating possible bias

    labels = []
    text = []
    for tweet in tweets:
        data = tweet.split(',')
        text.append(data[0])
        labels.append(data[1])

    # Dividing dataset into training and testing.
    test_size = int(len(labels)*test_percentage)

    x_train = text[:test_size]
    y_train = labels[:test_size]
    x_test = text[test_size:]
    y_test = labels[test_size:]
        
    return x_train, y_train, x_test, y_test


def load_word_dictionary(file):
    d = dict
    count = 1 # Starts with one since 0 is for unknown
    with open(file, 'r') as f:
        for word in f.readlines():
            d[word] = count
            count += 1
    return dict


def create_word_dictionary(sentences, thresh=5):
    count = dict()
    words = dict()

    for sentence in sentences:
        w = re.sub(r'[^a-zA-Z0-9\s]', '', sentence).split()
        w = [word.lower() for word in w]
        for word in w:
            if word in count.keys():
                count[word] += 1
            else:
                count[word] = 1

    valid_words = [word for word in count.keys() if count[word] > thresh]

    i = 1
    for word in valid_words:
        words[word] = i
        i += 1

    return words


def save_dict(dictionary):
    f = open('dict.txt', 'w')
    for word in dictionary.keys():
        f.write(word + '\n')
    f.close()


def get_max_sentence_size(sentences):
    max_sentence = 0
    for sentence in sentences:
        if max_sentence < len(sentence.split()):
            max_sentence = len(sentence.split())
    return max_sentence


def translate_to_tokens(sentences, maxSize, dictionary):
    results = []
    for sentence in sentences:
        seq = [0] * maxSize #0 = unknown
        sentence = re.sub(r'[^a-zA-Z0-9\s]', '', sentence).split()
        i = 0
        for word in sentence:
            start = maxSize-len(sentence)
            if word.lower() in dictionary.keys():
                seq[i+start] = dictionary[word]
            i += 1
        results.append(seq)

    return np.array(results)
