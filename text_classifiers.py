from random import shuffle
import re
import numpy as np
import tensorflow as tf
import keras as K

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding, Activation, LSTM
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
            self.dictionary = load_word_dictionary(dictionary)
            print("Dictionary loaded")

        # Tokenize data
        self.maxSentenceSize = get_max_sentence_size(self.x_train+self.x_test)
        self.x_train_tokens = translate_to_tokens(self.x_train, self.maxSentenceSize, self.dictionary)
        self.x_test_tokens = translate_to_tokens(self.x_test, self.maxSentenceSize, self.dictionary)

        print(self.x_train_tokens[10])
        print(self.y_train[10])
        print(self.x_train[10])

    def initiate_model(self, weights=None):
        print('Creating model...')
        words = len(self.dictionary)
        vec_len = 100
        model = Sequential()
        model.add(Embedding(input_dim=words, output_dim=vec_len, name='layer_embedding'))
        model.add(LSTM(units=50, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(units=3, activation='sigmoid'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        print(model.summary())
        # Training or loading
        if weights is None:
            print('Training model...')
            model.fit(self.x_train_tokens, np.array(self.y_train), validation_split=0.05, epochs=5, batch_size=32)
            print("Saving model...")
            model.save('sentiment_model1.h5')
        else:
            print('Loading weights...')
            model.load_weights(weights)
        self.model = model
        # Evaluation
        # loss = model.evaluate(np.array(self.x_test_tokenssd), np.array(self.y_test))
        # print(loss)

    def predict(self, text):
        return self.model.predict_classes(np.array(translate_to_tokens(text, self.maxSentenceSize, self.dictionary)))


class TextClassifierModel2:

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
            self.dictionary = load_word_dictionary(dictionary)
            print("Dictionary loaded")

        # Tokenize data
        self.maxSentenceSize = get_max_sentence_size(self.x_train+self.x_test)
        self.x_train_tokens = translate_to_tokens(self.x_train, self.maxSentenceSize, self.dictionary)
        self.x_test_tokens = translate_to_tokens(self.x_test, self.maxSentenceSize, self.dictionary)

        print(self.x_train_tokens[10])
        print(self.y_train[10])
        print(self.x_train[10])

    def initiate_model(self, weights=None):
        print('Creating model...')
        words = len(self.dictionary)
        vec_len = 100
        model = Sequential()
        model.add(Embedding(input_dim=words, output_dim=vec_len, name='layer_embedding'))
        model.add(LSTM(units=10, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(units=3, activation='sigmoid'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        print(model.summary())
        # Training or loading
        if weights is None:
            print('Training model...')
            model.fit(self.x_train_tokens, np.array(self.y_train), validation_split=0.05, epochs=5, batch_size=32)
            print("Saving model...")
            model.save('sentiment_model2.h5')
        else:
            print('Loading weights...')
            model.load_weights(weights)
        self.model = model
        # Evaluation
        # loss = model.evaluate(np.array(self.x_test_tokenssd), np.array(self.y_test))
        # print(loss)

    def predict(self, text):
        return self.model.predict_classes(np.array(translate_to_tokens(text, self.maxSentenceSize, self.dictionary)))


def load_training_data(file, test_percentage=0.2):
    '''
    Loads data from csv and returns the training/testing sets.
    :param file: comma separated file. First column is text and second one is its classification.
    :param test_percentage: percentage of data for the test set.
    :return: trainingSet,
    '''

    with open(file, 'r') as f:
        tweets = f.readlines()

    shuffle(tweets) # Mitigating possible bias

    labels = []
    text = []
    for tweet in tweets:
        data = tweet.split(',')
        text.append(data[0])
        labels.append(int_to_vec(int(data[1].strip())))

    # Dividing dataset into training and testing.
    test_size = int(len(labels)*test_percentage)

    x_train = text[:test_size]
    y_train = labels[:test_size]
    x_test = text[test_size:]
    y_test = labels[test_size:]
        
    return x_train, y_train, x_test, y_test


def int_to_vec(num, max=3):
    out = []
    for i in range(max):
        if num == i:
            out.append(1)
        else:
            out.append(0)
    return out


def load_word_dictionary(file):
    #TODO undefined values, start/end of tweet
    d = {}
    count = 1 # Starts with one since 0 is for unknown
    with open(file, 'r') as f:
        for word in f.readlines():
            d[word] = count
            count += 1
    return dict


def create_word_dictionary(sentences, thresh=5):
    count = {}
    words = {}

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


def translate_to_tokens(sentences, max_size, dictionary):
    results = []
    for sentence in sentences:
        seq = [0] * max_size #0 = unknown
        sentence = re.sub(r'[^a-zA-Z0-9\s]', '', sentence).split()
        sentence = [word.lower() for word in sentence]
        i = 0
        for word in sentence:
            start = max_size-len(sentence)
            if word.lower() in dictionary.keys():
                seq[i+start] = dictionary[word]
            i += 1
        results.append(seq)

    return np.array(results)
