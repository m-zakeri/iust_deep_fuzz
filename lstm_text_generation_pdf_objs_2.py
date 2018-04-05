"""
PDF OBJ 2
Train with generator for large datasets.
My strategy in this script is departed and not work,
so I decide to build new script.
Please refer to PDF OBJ 3 (lstm_text_generation_pdf_objs_3.py)


"""

from __future__ import print_function

from docutils.nodes import paragraph
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback
from keras.utils import plot_model
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io

import datetime
import pdf_object_preprocess as preprocess


def save(model, epochs):
    """ save the model and the plot """
    dt = datetime.datetime.now().strftime('_%Y%m%d_%H%M%S_')
    save_name = 'lstm_text_generation_pdf_objs_1' + dt + 'epochs' + str(epochs)
    # plot the model
    plot_model(model, to_file='./modelpic/' + save_name + '.png',
               show_shapes=True, show_layer_names=True)
    # Save model (requires HDF5 and h5py)
    model.save('./modelh5/' + save_name + '.h5')


def dataset2npyfilse():
    """ for using with data generator"""
    trainset_path = './trainset/pdfobjs.txt'
    trainset_path = './trainset/pdf_object_trainset_100_to_500_percent10.txt'
    text = preprocess.load_from_file(trainset_path)
    print('corpus length:', len(text))

    chars = sorted(list(set(text)))
    print('Total chars:', len(chars))
    # print(chars)

    # Vectorization
    print('Building dictionary index ...')
    char_indices = dict((c, i) for i, c in enumerate(chars))
    # print(char_indices)
    indices_char = dict((i, c) for i, c in enumerate(chars))
    # print(indices_char)

    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = 100  # Good idea: use ave_object_len to determine this hyper-parameter
    step = 1  # should set to 1 for best result
    epochs = 10  # number of epochs for training
    sentences = []  # list of all sentence as input
    next_chars = []  # list of all next chars as labels
    for i in range(0, len(text) - maxlen, step):  # arg2 why this?
        sentences.append(text[i: i + maxlen])
        preprocess.save_to_file('./npysamples/IDs/id-' + str(i), text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
        preprocess.save_to_file('./npysamples/Labels/id-' + str(i), text[i + maxlen])

    print('semi sequences:', len(sentences))

    print('end...')

"""
    for i, sentence in enumerate(sentences):
        x = np.zeros((1, maxlen, len(chars)), dtype=np.bool)  # input x
        y = np.zeros((1, len(chars)), dtype=np.bool)  # output label y
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1
        y[0, char_indices[next_chars[i]]] = 1
        np.save('./npysamples/IDs/id-' + str(i), x)
        np.save('./npysamples/Labels/id-' + str(i), y)
"""


def train():
    trainset_path = './trainset/pdfobjs.txt'
    trainset_path = './trainset/pdf_object_trainset_100_to_500_percent20.txt'
    text = preprocess.load_from_file(trainset_path)
    print('corpus length:', len(text))

    chars = sorted(list(set(text)))
    print('Total chars:', len(chars))
    # print(chars)

    # Vectorization
    print('Building dictionary index ...')
    char_indices = dict((c, i) for i, c in enumerate(chars))
    # print(char_indices)
    indices_char = dict((i, c) for i, c in enumerate(chars))
    # print(indices_char)

    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = 100  # Good idea: use ave_object_len to determine this hyper-parameter
    step = 1  # should set to 1 for best result
    epochs = 10  # number of epochs for training
    sentences = []  # list of all sentence as input
    next_chars = []  # list of all next chars as labels
    for i in range(0, len(text) - maxlen, step):  # arg2 why this?
        sentences.append(text[i: i + maxlen])
        # print(sentences)
        next_chars.append(text[i + maxlen])
        # print(next_chars)

    print('semi sequences:', len(sentences))
    
    divide_factor = int(1e6)  # 1m
    partial_seq_len = int(len(sentences) / divide_factor)
    X = [] # input list
    Y = [] # output list
    print('One-Hot vectorization...')
    print(partial_seq_len)
    for z in range(partial_seq_len):
        x = (np.zeros((divide_factor, maxlen, len(chars)), dtype=np.bool))  # input x
        y = (np.zeros((divide_factor, len(chars)), dtype=np.bool))  # output label y
        for i, sentence in enumerate(sentences[z*divide_factor:(z+1)*divide_factor]):
            for t, char in enumerate(sentence):
                x[i, t, char_indices[char]] = 1
            y[i, char_indices[next_chars[i]]] = 1
        X.append(x)
        Y.append(y)

    v = len(sentences) - partial_seq_len * divide_factor
    x = (np.zeros((v, maxlen, len(chars)), dtype=np.bool))  # input x
    y = (np.zeros((v, len(chars)), dtype=np.bool))  # output label y
    for i, sentence in enumerate(sentences[partial_seq_len * divide_factor:]):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
    X.append(x)
    Y.append(y)

    # build the model: a single LSTM layer # we need to deep it
    print('Build model...')
    model = Sequential()
    # model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(LSTM(128, input_shape=(maxlen, len(chars)), return_sequences=True))
    # model.add(LSTM(128, input_shape=(maxlen, len(chars)),
    #                activation='relu', return_sequences=True, dropout=0.2))
    model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    # model.add(LSTM(128, activation='relu', dropout=0.2))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    # sys.exit()

    print('fitfit', partial_seq_len)
    for i in range(partial_seq_len):
        model.fit(X[i], Y[i], batch_size=128, epochs=epochs, validation_split=0.2)  # why epochs=?

    save(model, epochs)

    # del model
    # model = load_model('./modelh5/lstm_text_generation_pdf_objs_1_20180214_235713_epochs10.h5')

    """ sampling the model and generate new object """
    # configuration parameters
    diversities = [0.2, 0.5, 1.0, 1.2, 1.5, 1.8]
    # diversities = [i*0.10 for i in range(1,20,2)]
    generated_obj_max_number = 5
    generated_obj_max_allowed_len = 500
    t_fuzz = 0.9
    p_t = 0.9  # 0.9 for format fuzzing and 0.5 or letter for data fuzzing. Now format fuzzing
    # end of configuration parameters

    list_of_objects = preprocess.get_list_of_object(text)
    list_of_objects_with_maxlen = []
    for o in list_of_objects:
        if len(o) > maxlen:
            list_of_objects_with_maxlen.append(o)

    for diversity in diversities:
        obj_index = random.randint(0, len(list_of_objects_with_maxlen) - 1)
        generated_obj_counter = 0
        generated_obj_len_index = 0
         
        stop_condition = False
        print()
        print('-- Diversity:', diversity)

        # generated = ''
        obj_prefix = str(list_of_objects_with_maxlen[obj_index])[0: maxlen]  # len(sentence) equals 100 here

        generated = obj_prefix
        prob_vals = '100\n' * maxlen
        learnt_grammar = obj_prefix

        print('--- Generating text with seed:\n "' + obj_prefix + '"')
        sys.stdout.write(generated)

        if generated.endswith('endobj'):
            generated_obj_counter += 1

        if generated_obj_counter > generated_obj_max_number:
            stop_condition = True

        while not stop_condition:
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(obj_prefix):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index, prob, preds2 = sample(preds, diversity)
            p_fuzz = random.random()
            if p_fuzz > t_fuzz and preds2[next_index] > p_t:
                next_index = np.argmin(preds2)
                print('FUZZ DONE!')

            next_char = indices_char[next_index]

            # print()
            # print(preds2)
            # print(np.argmax(preds))
            # print(preds[np.argmax(preds)])

            # print(prob)
            # print(np.argmax(prob))
            # print('====>',next_index)
            # print(prob[0, next_index])

            # prob_vals += str(preds2[next_index]) + '\n'
            # if preds2[next_index] > 0.9980:
            #     learnt_grammar += next_char
            # else:
            #     learnt_grammar += '.'
            # input()

            obj_prefix = obj_prefix[1:] + next_char
            generated += next_char
            generated_obj_len_index += 1

            if generated.endswith('endobj'):
                generated_obj_counter += 1
                generated_obj_len_index = 0
            elif generated_obj_len_index > generated_obj_max_allowed_len:
                generated += '\nendobj\n'
                generated_obj_counter += 1
                generated_obj_len_index = 0

            if generated_obj_counter > generated_obj_max_number:
                stop_condition = True

            sys.stdout.write(next_char)
            sys.stdout.flush()

        # save generated text to file inside program
        dt = datetime.datetime.now().strftime('_%Y%m%d_%H%M%S_')
        dir_name = './generated_results/pdfobjs_new/'
        file_name = 'gen_objs' + dt + 'epochs' + repr(epochs) + '_div' \
                    + repr(diversity) + '_step' + repr(step) + '.txt'
        preprocess.save_to_file(dir_name + file_name, generated)
        # poc.save_to_file(dir_name + file_name + 'probabilities.txt', prob_vals)
        # poc.save_to_file(dir_name + file_name + 'learntgrammar.txt',learnt_grammar)


# lower temperature will cause the model to make more likely,
# but also more boring and conservative predictions.
def sample(preds, temperature=1.0):
    """helper function to sample an index from a probability array"""
    # print('raw predictions = ', preds)
    preds = np.asarray(preds).astype('float64')

    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    """sampling:"""
    probas = np.random.multinomial(1, preds, 1)
    # print()
    # print('sanitize predictions = ', preds)
    return np.argmax(probas), probas, preds


def no_sample():
    pass


def sample_space():
    pass


def main(argv):
    """ the main function """
    # train()
    dataset2npyfilse()

if __name__ == "__main__":
    main(sys.argv)
